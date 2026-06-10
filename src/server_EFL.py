# server_EFL.py
#
# Event-Driven Continual Federated Learning — Server (Algorithms 2, 3, 6)
#
# Server responsibilities
# ───────────────────────
#   Phase 2a  — Pool Admission via MMD           (Algorithm 2)
#   Phase 2b  — Teacher Training on pool data    (Algorithm 3)
#   Full loop — Orchestrate all phases           (Algorithm 6)
#
# Privacy guarantee:
#   Only VAE generator weights cross the client→server boundary.
#   The server never receives raw traffic or gradient tensors.
#   The Teacher is broadcast server→client; no client model is aggregated.

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Protocol, runtime_checkable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (f1_score, confusion_matrix,
                             classification_report)

from client_EFL import ClientEFL
from models import TeacherModel, ZIRVAEDecoder
from utils import save_results_as_json


@runtime_checkable
class Generator(Protocol):
    """Any object that can generate synthetic samples."""
    def generate(self, n: int, device: str) -> np.ndarray: ...


# ────────────────────────────────────────────────────────────────────────────
# MMD helpers  (Definition 4 in the EFL spec)
# ────────────────────────────────────────────────────────────────────────────

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """κ(x, y) = exp(−‖x−y‖² / 2σ²)  evaluated pairwise over rows."""
    diff = X[:, None, :] - Y[None, :, :]      # (P, P, D)
    sq   = (diff ** 2).sum(axis=-1)           # (P, P)
    return np.exp(-sq / (2.0 * sigma ** 2))


def _median_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
    """Set σ via the median heuristic on the pooled samples."""
    Z    = np.vstack([X, Y])
    diff = Z[:, None, :] - Z[None, :, :]
    sq   = (diff ** 2).sum(axis=-1)
    return float(np.sqrt(np.median(sq[sq > 0]) / 2.0 + 1e-8))


def unbiased_mmd2(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased empirical MMD² (Definition 4).

    X, Y : (P, D)  two sample sets of equal size P.
    Returns a scalar ≥ 0 (may be slightly negative due to finite-sample
    unbiasedness — clamp to 0 when used as a distance).
    """
    P     = X.shape[0]
    sigma = _median_bandwidth(X, Y)

    Kxx   = _rbf_kernel(X, X, sigma)
    Kyy   = _rbf_kernel(Y, Y, sigma)
    Kxy   = _rbf_kernel(X, Y, sigma)

    # zero out diagonal for unbiased estimator
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term1 = Kxx.sum() / (P * (P - 1))
    term2 = Kyy.sum() / (P * (P - 1))
    term3 = 2.0 * Kxy.sum() / (P ** 2)

    return max(0.0, term1 + term2 - term3)


# ────────────────────────────────────────────────────────────────────────────
# Fingerprint helpers  (Definition 3)
# ────────────────────────────────────────────────────────────────────────────

def _draw_fingerprint(decoder, P: int, device: str) -> np.ndarray:
    """
    Draw P attack-only synthetic samples for MMD fingerprinting.
    For _ConditionalTabDDPM uses attack decoder (class=1) only so
    MMD compares attack distributions across clients.
    Returns (P, T*F) float32 — prototype fingerprint φ^(k).
    """
    if hasattr(decoder, 'models') and 1 in decoder.models:
        samples = decoder.models[1].generate(n=P, device=device)
    else:
        samples = decoder.generate(n=P, device=device)
    return samples.reshape(P, -1).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Pool management  (Algorithms 2 & 3)
# ────────────────────────────────────────────────────────────────────────────

class GeneratorPool:
    """
    Server-side generator pool P = {G_0, G_1, …, G_K}.

    Holds ZIRVAEDecoder objects — only the decoder is received from clients.
    The encoder (which processed real traffic) stays on the client.

    Attributes
    ----------
    generators   : list[ZIRVAEDecoder]  — indexed by pool label k
    fingerprints : list[np.ndarray]     — φ^(k), shape (P_fp, T*F)
    labels       : list[str]            — human-readable tag per generator
    mmd_log      : list[dict]           — admission decision log for all rounds
    """

    def __init__(self, fingerprint_size: int = 100, tau_merge: float = 0.05,
                 device: str = 'cpu'):
        self.fingerprint_size = fingerprint_size
        self.tau_merge        = tau_merge
        self.device           = device

        self.generators  : list = []
        self.fingerprints: list[np.ndarray] = []
        self.labels      : list[str] = []
        self.mmd_log     : list[dict] = []

    @property
    def K(self) -> int:
        """Number of entries currently in the pool (including benign G_0)."""
        return len(self.generators)

    def add_initial(self, decoder: Generator, label: str = 'benign_G0'):
        """Register the initial benign decoder at boot time (skips MMD check)."""
        fp = _draw_fingerprint(decoder, self.fingerprint_size, self.device)
        self.generators.append(decoder)
        self.fingerprints.append(fp)
        self.labels.append(label)
        print(f"  [Pool] Initial decoder registered: '{label}'  K={self.K}")

    def try_admit(self, new_dec: Generator, label: str = '') -> bool:
        """
        Phase 2a (Algorithm 2): MMD-based admission gate.

        Steps
        -----
        1. Draw fingerprint φ_new from new_dec (P samples from N(0,I)).
        2. Compute MMD²(φ_new, φ_k) for every existing pool entry k.
        3. If min MMD² < τ_merge → discard (this distribution is already covered).
        4. Otherwise admit: add decoder + fingerprint to pool, K ← K+1.

        Returns True if admitted, False if discarded.
        """
        fp_new = _draw_fingerprint(new_dec, self.fingerprint_size, self.device)
        tag    = label if label else f'G_{self.K}'

        mmds = {}
        for k, fp_k in enumerate(self.fingerprints):
            mmds[self.labels[k]] = round(float(unbiased_mmd2(fp_new, fp_k)), 6)

        min_mmd  = min(mmds.values()) if mmds else float('inf')
        admitted = min_mmd >= self.tau_merge

        self.mmd_log.append({
            'label'    : tag,
            'mmd_vs'   : mmds,
            'min_mmd2' : min_mmd,
            'tau_merge': self.tau_merge,
            'admitted' : admitted,
        })

        if not admitted:
            closest = min(mmds, key=mmds.get)
            print(f"  [Pool 2a] DISCARD '{tag}'  "
                  f"min_MMD²={min_mmd:.6f} < τ_merge={self.tau_merge}  "
                  f"(closest: '{closest}')")
            return False

        self.generators.append(new_dec)
        self.fingerprints.append(fp_new)
        self.labels.append(tag)
        print(f"  [Pool 2a] ADMIT   '{tag}'  "
              f"min_MMD²={min_mmd:.6f} ≥ τ_merge={self.tau_merge}  "
              f"new K={self.K}")
        return True

    def sample_synthetic_dataset(self, n_per_gen: int, device: str
                                 ) -> tuple[np.ndarray, np.ndarray]:
        """
        Phase 2b: draw N_s attack-only samples per pool generator.

        For _ConditionalTabDDPM generators, samples only from the attack
        decoder (class label 1) so the Teacher learns to distinguish
        attack types, not benign vs attack.

        Returns (X, y) where y ∈ {0, …, K-1} = pool index (attack type).
        """
        all_X, all_y = [], []
        for k, gen in enumerate(self.generators):
            # _ConditionalTabDDPM: sample attack class only
            if hasattr(gen, 'models') and 1 in gen.models:
                X_k = gen.models[1].generate(n=n_per_gen, device=device)
                src = 'attack decoder'
            else:
                # ZI-RVAE or other — generate from full model
                X_k = gen.generate(n=n_per_gen, device=device)
                src = 'full generator'
            all_X.append(X_k.astype(np.float32))
            all_y.append(np.full(n_per_gen, k, dtype=np.int64))
            print(f"    [Pool 2b] '{self.labels[k]}' ({src})"
                  f"→ {n_per_gen} samples  shape={X_k.shape}  (attack_type={k})")

        return np.concatenate(all_X, axis=0), np.concatenate(all_y)


# ────────────────────────────────────────────────────────────────────────────
# MMD matrix printer
# ────────────────────────────────────────────────────────────────────────────

def _print_mmd_matrix(fps: list[np.ndarray], labels: list[str]):
    """
    Print the full N×N pairwise MMD² matrix for a list of decoder fingerprints.

    fps    : list of (P, D) fingerprint arrays — one per client decoder
    labels : matching short label per decoder (e.g. 'C0', 'C1', …)

    Diagonal is 0 by definition (same distribution vs itself).
    Called once after all clients push decoders, before any admission decision.
    """
    n = len(fps)
    if n == 0:
        return

    # pre-compute upper triangle, mirror to lower
    mmd_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = unbiased_mmd2(fps[i], fps[j])
            mmd_mat[i, j] = v
            mmd_mat[j, i] = v

    col_w  = 12
    short  = [lbl[:col_w] for lbl in labels]
    header = f"{'':18s}" + "".join(f"{s:>{col_w}}" for s in short)
    sep    = "  " + "-" * len(header)

    print(f"\n  {'Pairwise MMD² Matrix  (N×N, all client decoders)':^{len(header)}}")
    print(sep)
    print(f"  {header}")
    print(sep)
    for i, (row_fps, lbl) in enumerate(zip(mmd_mat, short)):
        row = f"  {lbl:<18}" + "".join(f"{v:>{col_w}.6f}" for v in row_fps)
        print(row)
    print(sep + "\n")

    return mmd_mat


# ────────────────────────────────────────────────────────────────────────────
# TSTR / TRTS summary table
# ────────────────────────────────────────────────────────────────────────────

def _print_tstr_trts_summary(client_list: list):
    """
    Print a formatted table of TSTR/TRTS/MMD² for both ZI-RVAE and TabDDPM.
    """
    rows = []
    for c in client_list:
        for domain, entry in c.generators.items():
            rows.append((c.client_id, domain, entry))

    if not rows:
        return

    col = 10
    hdr = (f"  {'Client':>8}  {'Domain':<24}  {'Generator':<10}"
           f"{'TSTR f1':>{col}}{'TRTS f1':>{col}}{'MMD²':>{col+2}}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(f"\n  {'Generator Quality — ZI-RVAE vs TabDDPM':^{len(hdr)-2}}")
    print(sep)
    print(hdr)
    print(sep)
    for cid, dom, entry in rows:
        for gen_name, tstr_key, trts_key, mmd_key in [
            ('ZI-RVAE',  'tstr',      'trts',      'mmd2'),
            ('TabDDPM',  'tstr_ddpm', 'trts_ddpm', 'mmd2_ddpm'),
        ]:
            tstr = entry.get(tstr_key, {})
            trts = entry.get(trts_key, {})
            mmd2 = entry.get(mmd_key, float('nan'))
            print(f"  {cid:>8}  {dom:<24}  {gen_name:<10}"
                  f"{tstr.get('f1_macro', 0):>{col}.4f}"
                  f"{trts.get('f1_macro', 0):>{col}.4f}"
                  f"{mmd2:>{col+2}.6f}")
    print(sep + "\n")


# ────────────────────────────────────────────────────────────────────────────
# Teacher training + evaluation  (Phase 2b / Algorithm 3)
# ────────────────────────────────────────────────────────────────────────────

def train_teacher(teacher: TeacherModel, pool: 'GeneratorPool',
                  n_per_gen: int, device: torch.device,
                  epochs: int = 100, lr: float = 1e-3,
                  batch_size: int = 256,
                  train_ratio: float = 0.75) -> tuple:
    """
    Phase 2b (Algorithm 3):
      1. Draw N_s synthetic samples per pool decoder → full dataset D'.
      2. Split 75 % train / 25 % test (stratified by class label).
      3. Train Teacher on the train split with CrossEntropy.
      4. Evaluate on the test split: F1 (macro + per-class) + confusion matrix.

    Returns (teacher, eval_dict) where eval_dict contains:
        f1_macro, f1_per_class, confusion_matrix, classification_report
    """
    K = pool.K
    teacher.grow_to_k(K)
    teacher = teacher.to(device)

    # ── 1. Generate full synthetic dataset ────────────────────────────
    X_np, y_np = pool.sample_synthetic_dataset(n_per_gen, str(device))

    # ── 2. Stratified 75 / 25 split ───────────────────────────────────
    rng       = np.random.default_rng(seed=42)
    tr_idx, te_idx = [], []
    for cls in np.unique(y_np):
        idx  = np.where(y_np == cls)[0]
        idx  = rng.permutation(idx)
        n_tr = max(1, int(len(idx) * train_ratio))
        tr_idx.extend(idx[:n_tr].tolist())
        te_idx.extend(idx[n_tr:].tolist())

    X_tr, y_tr = X_np[tr_idx], y_np[tr_idx]
    X_te, y_te = X_np[te_idx], y_np[te_idx]

    print(f"    [Teacher 2b] Dataset: {len(X_np)} total  "
          f"→ train {len(X_tr)} / test {len(X_te)}  "
          f"(classes: {np.unique(y_np).tolist()})")

    # ── 3. Train ──────────────────────────────────────────────────────
    # X_tr is (N, T, F) — LSTM Teacher consumes sequences directly
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)   # (N, T, F)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    loader  = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                         batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(teacher.parameters(), lr=lr)
    ce        = nn.CrossEntropyLoss()
    teacher.train()

    bar = tqdm(range(epochs), desc="Teacher 2b", leave=True, dynamic_ncols=True)
    for _ in bar:
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = ce(teacher(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        bar.set_postfix(loss=f"{ep_loss/max(len(loader),1):.4f}", K=K)

    # ── 4. Evaluate on held-out 25 % ──────────────────────────────────
    teacher.eval()
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)  # (N, T, F)
    with torch.no_grad():
        logits = teacher(X_te_t)
        preds  = torch.argmax(logits, dim=1).cpu().numpy()

    class_names = [pool.labels[k] for k in range(K)]

    f1_macro    = float(f1_score(y_te, preds, average='macro',
                                 zero_division=0))
    f1_per_cls  = f1_score(y_te, preds, average=None,
                           zero_division=0).tolist()
    cm          = confusion_matrix(y_te, preds).tolist()
    report      = classification_report(y_te, preds,
                                        target_names=class_names,
                                        zero_division=0)

    # ── 5. Pretty-print results ───────────────────────────────────────
    print(f"\n  {'─'*56}")
    print(f"  Teacher (Phase 2b) Evaluation  —  test set n={len(y_te)}")
    print(f"  {'─'*56}")
    print(f"  F1 macro : {f1_macro:.4f}")
    print(f"\n  F1 per class:")
    for lbl, f1v in zip(class_names, f1_per_cls):
        print(f"    {lbl:<30}  f1={f1v:.4f}")
    print(f"\n  Confusion matrix  (rows=true, cols=pred):")
    header_cm = "  " + " " * 22 + "".join(f"{l[:8]:>10}" for l in class_names)
    print(header_cm)
    for row_cm, lbl in zip(cm, class_names):
        print(f"  {lbl[:20]:<22}" + "".join(f"{v:>10}" for v in row_cm))
    print(f"\n  Classification report:")
    for line in report.splitlines():
        print(f"    {line}")
    print(f"  {'─'*56}\n")

    eval_dict = {
        'f1_macro'              : f1_macro,
        'f1_per_class'          : dict(zip(class_names, f1_per_cls)),
        'confusion_matrix'      : cm,
        'class_names'           : class_names,
        'n_train'               : int(len(y_tr)),
        'n_test'                : int(len(y_te)),
    }

    return teacher, eval_dict


# ────────────────────────────────────────────────────────────────────────────
# Full system loop  (Algorithm 6)
# ────────────────────────────────────────────────────────────────────────────

def server_efl(args, device, domains_path,
               client_distributions, max_client_participants,
               tot_time_steps, project_root,
               tau_merge:      float = 0.05,
               fingerprint_P:  int   = 100,
               n_per_gen:      int   = 300,
               teacher_epochs: int   = 100):
    """
    EFL main loop — attack generators only.

    Phase 1 : All N clients independently train a ZI-RVAE on their
              local attack traffic and push only the decoder to the server.
    Phase 2a: Server fingerprints every decoder, prints the N×N pairwise
              MMD² matrix, then runs the MMD admission gate to build the
              generator pool (duplicate distributions are discarded).

    Phases 2b, 3, 4 (Teacher training, Watcher, Distilled Replay) are
    not yet enabled — this run stops after pool construction.
    """
    print("\n=== Event-Driven Continual FL (EFL) — attack generator phase ===")

    # MPS has a hard memory ceiling (~20 GB shared with OS) and is not
    # thread-safe. ZI-RVAE training is GRU-based and runs fast on CPU.
    # Force CPU for EFL to avoid OOM and segfaults on Apple Silicon.
    if str(device) == 'mps':
        print("  [EFL] MPS detected — forcing CPU for stable ZI-RVAE training.")
        device = torch.device('cpu')

    attacktype = getattr(args, 'attacktype', None) or 'all'
    models_dir = os.path.join(project_root, 'saved_models_efl', attacktype)
    os.makedirs(models_dir, exist_ok=True)
    print(f"  [EFL] Attack type : {attacktype}")
    print(f"  [EFL] Models dir  : {models_dir}")

    # ── Generator pool ────────────────────────────────────────────────
    pool = GeneratorPool(
        fingerprint_size = fingerprint_P,
        tau_merge        = tau_merge,
        device           = str(device),
    )

    # ── Load previously saved pool decoders (resume across runs) ─────
    pool_files = sorted(f for f in os.listdir(models_dir)
                        if f.startswith('pool_k') and f.endswith('.pt'))
    if pool_files:
        print(f"\n  [Server] Found {len(pool_files)} saved pool decoder(s) — loading…")
        for fname in pool_files:
            ckpt = torch.load(os.path.join(models_dir, fname),
                              map_location=device)
            gen_type = ckpt.get('gen_type', 'zirvae')
            if gen_type == 'tabddpm':
                from models import TabDDPMGenerator
                from client_EFL import _ConditionalTabDDPM
                models_map = {}
                for cls_str, denoiser_sd in ckpt['denoiser_per_class'].items():
                    g = TabDDPMGenerator(
                        data_dim    = ckpt['data_dim'],
                        window_size = ckpt['window_size'],
                        n_features  = ckpt['n_features'],
                        n_steps     = ckpt['n_steps'],
                    ).to(device)
                    g.denoiser.load_state_dict(denoiser_sd)
                    g.eval()
                    models_map[int(cls_str)] = g
                gen = _ConditionalTabDDPM(models_map)
            else:
                gen = ZIRVAEDecoder(
                    latent_dim    = ckpt['latent_dim'],
                    hidden_dim    = ckpt['hidden_dim'],
                    n_features    = ckpt['n_features'],
                    window_size   = ckpt['window_size'],
                    feat_type_idx = ckpt['feat_type_idx'],
                    n_layers      = ckpt['n_layers'],
                ).to(device)
                gen.load_state_dict(ckpt['state_dict'])
            gen.eval()
            label = ckpt.get('pool_label', fname.replace('.pt', ''))
            pool.add_initial(gen, label=label)
            print(f"  [Server] Loaded pool {gen_type} '{label}' "
                  f"(k={ckpt.get('pool_index', '?')}) ← {fname}")
        print(f"  [Server] Pool restored — K={pool.K}  "
              f"labels={pool.labels}\n")
    else:
        print("  [Server] No saved pool decoders found — starting fresh.\n")

    # ── Teacher (server-side LSTM, grows with pool) ───────────────────
    input_dim = args.window_size * args.n_raw_features
    teacher = TeacherModel(
        input_dim   = input_dim,
        hidden_dim  = 128,
        n_classes   = 1,
        n_features  = args.n_raw_features,
        window_size = args.window_size,
    ).to(device)

    pool_size_log:    list[int] = []
    generator_quality: dict     = {}
    mmd_matrices:      dict     = {}
    teacher_eval_log:  dict     = {}
    student_eval_log:  dict     = {}

    # ── helper: Phase 4 distilled replay for all clients ─────────────
    def _run_phase4(client_list, teacher, pool, tag: str):
        """
        For each client:
          1. Train baseline Student (real data only, no distillation).
          2. Train distilled Student (real + synthetic + Teacher KD).
          3. Evaluate both and print side-by-side comparison.
        """
        print(f"\n  --- Phase 4: Baseline + Distilled Replay  "
              f"({len(client_list)} clients) ---")
        for c in client_list:
            # ── baseline ─────────────────────────────────────────────
            print(f"\n  [Client {c.client_id}] Training Baseline Student …")
            c.train_student_baseline(epochs=30, lr=1e-3)
            base_eval = c.evaluate_student_baseline()
            student_eval_log[f"{tag}_client{c.client_id}_baseline"] = base_eval

            # ── distilled ────────────────────────────────────────────
            print(f"\n  [Server → Client {c.client_id}] Broadcasting Teacher …")
            c.receive_teacher(teacher)
            c.distilled_replay_update(pool.generators, n_syn=200,
                                      epochs=30, lr=1e-3,
                                      lambda2=1.0, tau=3.0)
            distil_eval = c.evaluate_student()
            student_eval_log[f"{tag}_client{c.client_id}_distilled"] = distil_eval

            # ── side-by-side comparison ───────────────────────────────
            _print_comparison(c.client_id, base_eval, distil_eval)

            # ── save distilled student checkpoint ─────────────────────
            ckpt_path = os.path.join(models_dir,
                                     f'student_{tag}_c{c.client_id}.pt')
            if hasattr(c, 'student') and c.student is not None:
                torch.save(c.student.state_dict(), ckpt_path)
                print(f"  [Client {c.client_id}] Distilled Student saved → {ckpt_path}")

    def _print_comparison(client_id: int, base: dict, distil: dict):
        """Print a side-by-side metric comparison table."""
        metrics = ['accuracy', 'f1_macro', 'f1_weighted']
        w = 12
        print(f"\n  {'─'*54}")
        print(f"  Client {client_id} — Baseline vs Distilled Comparison")
        print(f"  {'─'*54}")
        print(f"  {'Metric':<18} {'Baseline':>{w}} {'Distilled':>{w}} {'Δ':>{w}}")
        print(f"  {'─'*54}")
        for m in metrics:
            bv = base.get(m, 0.0)
            dv = distil.get(m, 0.0)
            delta = dv - bv
            sign  = '+' if delta >= 0 else ''
            print(f"  {m:<18} {bv:>{w}.4f} {dv:>{w}.4f} {sign}{delta:>{w-1}.4f}")
        print(f"  {'─'*54}\n")

    # ════════════════════════════════════════════════════════════════
    # MODE A: load saved pool decoders → Phase 2b only
    # ════════════════════════════════════════════════════════════════
    if getattr(args, 'load_decoders', False):
        if pool.K == 0:
            raise RuntimeError(
                "No pool decoders found in saved_models_efl/. "
                "Run without --load_decoders first to train and save them."
            )
        print(f"\n  [Server] --load_decoders mode: "
              f"skipping Phase 1 & 2a, using pool K={pool.K}")
        print(f"  Pool labels: {pool.labels}")

        print(f"\n  --- Phase 2b: Teacher training on pool K={pool.K} ---")
        teacher, teacher_eval = train_teacher(
            teacher    = teacher,
            pool       = pool,
            n_per_gen  = n_per_gen,
            device     = device,
            epochs     = teacher_epochs,
        )
        teacher_eval_log[0] = teacher_eval
        pool_size_log.append(pool.K)

        teacher_path = os.path.join(models_dir, f'teacher_K{pool.K}.pt')
        torch.save(teacher.state_dict(), teacher_path)
        print(f"  [Server] Teacher saved → {teacher_path}")

        # ── Phase 4: build clients and run distilled replay ───────────
        client_list_a: list[ClientEFL] = []
        for i in range(max_client_participants):
            c = ClientEFL(
                client_id        = i,
                args             = args,
                domain_path      = domains_path,
                assigned_domains = client_distributions[i],
                device           = device,
            )
            client_list_a.append(c)
        _run_phase4(client_list_a, teacher, pool, tag=f'K{pool.K}')

    # ════════════════════════════════════════════════════════════════
    # MODE B: full loop — Phase 1 → 2a → 2b each time step
    # ════════════════════════════════════════════════════════════════
    else:
        # ── Initialise clients ────────────────────────────────────────
        client_list: list[ClientEFL] = []
        for i in range(max_client_participants):
            c = ClientEFL(
                client_id        = i,
                args             = args,
                domain_path      = domains_path,
                assigned_domains = client_distributions[i],
                device           = device,
            )
            client_list.append(c)
        print(f"Initialised {len(client_list)} EFL clients.\n")

        for t in range(tot_time_steps):
            print(f"\n{'='*60}")
            print(f"  Time step {t + 1} / {tot_time_steps}")
            print(f"{'='*60}")

            # ── Phase 1 ───────────────────────────────────────────────
            use_parallel = str(device).startswith('cuda')
            mode = "parallel" if use_parallel else "sequential"
            print(f"\n  --- Phase 1: generator training "
                  f"({len(client_list)} clients, {mode}) ---")
            # (decoder_zi, tabddpm, label) per client
            pushed_decoders: list[tuple] = []

            def _train_one(c):
                if t >= len(c.domain_keys):
                    return None
                dec_zi, tabddpm = c.train_generator(time_step=t)
                label = f"C{c.client_id}_{c.domain_keys[t]}"
                return (c.client_id, dec_zi, tabddpm, label)

            if use_parallel:
                with ThreadPoolExecutor(max_workers=len(client_list)) as ex:
                    futures = {ex.submit(_train_one, c): c for c in client_list}
                    results_phase1 = {}
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res is None:
                            c = futures[fut]
                            print(f"  [Client {c.client_id}] no domain for t={t}, skipping.")
                            continue
                        cid, dec_zi, tabddpm, label = res
                        results_phase1[cid] = (dec_zi, tabddpm, label)
                        print(f"  [Client {cid}] decoders '{label}' pushed to server.")
                for c in client_list:
                    if c.client_id in results_phase1:
                        pushed_decoders.append(results_phase1[c.client_id])
            else:
                for c in client_list:
                    res = _train_one(c)
                    if res is None:
                        print(f"  [Client {c.client_id}] no domain for t={t}, skipping.")
                        continue
                    cid, dec_zi, tabddpm, label = res
                    pushed_decoders.append((dec_zi, tabddpm, label))
                    print(f"  [Client {cid}] decoders '{label}' pushed to server.")

            # save every pushed decoder
            for dec_zi, tabddpm, label in pushed_decoders:
                # ZI-RVAE decoder — only if trained
                if dec_zi is not None:
                    fname = os.path.join(models_dir, f"decoder_t{t}_{label}.pt")
                    torch.save({
                        'state_dict'   : dec_zi.state_dict(),
                        'latent_dim'   : dec_zi.latent_dim,
                        'hidden_dim'   : dec_zi.hidden_dim,
                        'n_features'   : dec_zi.n_features,
                        'window_size'  : dec_zi.window_size,
                        'n_layers'     : dec_zi.n_layers,
                        'feat_type_idx': dec_zi.feat_type_idx,
                    }, fname)
                    print(f"  [Server] ZI-RVAE decoder saved → {fname}")
                # TabDDPM — save only the denoiser weights (noise schedule is fixed math)
                if tabddpm is not None:
                    fname_ddpm = os.path.join(models_dir, f"tabddpm_t{t}_{label}.pt")
                    # _ConditionalTabDDPM: save per-class denoiser state dicts
                    class_denoisers = {}
                    for cls_label, gen in tabddpm.models.items():
                        class_denoisers[str(cls_label)] = gen.denoiser.state_dict()
                    torch.save({
                        'denoiser_per_class': class_denoisers,
                        'data_dim'          : tabddpm.data_dim,
                        'window_size'       : tabddpm.window_size,
                        'n_features'        : tabddpm.n_features,
                        'n_steps'           : tabddpm.n_steps,
                    }, fname_ddpm)
                    print(f"  [Server] TabDDPM denoiser saved → {fname_ddpm}")

            _print_tstr_trts_summary(client_list)
            for c in client_list:
                for domain, entry in c.generators.items():
                    key = f'client{c.client_id}_{domain}'
                    generator_quality[key] = {
                        'zirvae': {
                            'tstr'         : entry.get('tstr', {}),
                            'trts'         : entry.get('trts', {}),
                            'mmd2'         : entry.get('mmd2', None),
                            'recon_history': entry.get('recon_history', []),
                        },
                        'tabddpm': {
                            'tstr'         : entry.get('tstr_ddpm', {}),
                            'trts'         : entry.get('trts_ddpm', {}),
                            'mmd2'         : entry.get('mmd2_ddpm', None),
                            'loss_history' : entry.get('ddpm_loss_history', []),
                        },
                    }

            if not pushed_decoders:
                print("  No decoders pushed — skipping Phase 2a/2b.")
                pool_size_log.append(pool.K)
                continue

            # ── Phase 2a ──────────────────────────────────────────────
            print(f"\n  --- Phase 2a: MMD pool admission "
                  f"({len(pushed_decoders)} decoders) ---")
            # use whichever generator is available for fingerprinting
            fps    = [_draw_fingerprint(dec_zi if dec_zi is not None else tabddpm,
                                        fingerprint_P, str(device))
                      for dec_zi, tabddpm, _ in pushed_decoders]
            labels = [label for _, _, label in pushed_decoders]
            mmd_mat = _print_mmd_matrix(fps, labels)
            mmd_matrices[t] = {
                'labels': labels,
                'matrix': mmd_mat.tolist() if mmd_mat is not None else [],
            }

            for dec_zi, tabddpm, label in pushed_decoders:
                gen = dec_zi if dec_zi is not None else tabddpm
                if pool.K == 0:
                    pool.add_initial(gen, label=label)
                else:
                    pool.try_admit(gen, label=label)

            print(f"\n  Phase 2a complete — pool K={pool.K}  labels={pool.labels}")

            # save admitted pool generators (ZI-RVAE or TabDDPM)
            for k, (gen, lbl) in enumerate(zip(pool.generators, pool.labels)):
                fname   = os.path.join(models_dir, f"pool_k{k}_{lbl}.pt")
                is_ddpm = hasattr(gen, 'data_dim')   # _ConditionalTabDDPM
                if is_ddpm:
                    # save only denoiser weights per class — noise schedule is fixed
                    class_denoisers = {str(cls): m.denoiser.state_dict()
                                       for cls, m in gen.models.items()}
                    ckpt = {
                        'denoiser_per_class': class_denoisers,
                        'data_dim'          : gen.data_dim,
                        'window_size'       : gen.window_size,
                        'n_features'        : gen.n_features,
                        'n_steps'           : gen.n_steps,
                        'pool_label'        : lbl,
                        'pool_index'        : k,
                        'gen_type'          : 'tabddpm',
                    }
                else:
                    ckpt = {
                        'state_dict'   : gen.state_dict(),
                        'latent_dim'   : gen.latent_dim,
                        'hidden_dim'   : gen.hidden_dim,
                        'n_features'   : gen.n_features,
                        'window_size'  : gen.window_size,
                        'n_layers'     : gen.n_layers,
                        'feat_type_idx': gen.feat_type_idx,
                        'pool_label'   : lbl,
                        'pool_index'   : k,
                        'gen_type'     : 'zirvae',
                    }
                torch.save(ckpt, fname)
                print(f"  [Server] Pool generator saved → {fname}")

            pool_size_log.append(pool.K)

            # ── Phase 2b ──────────────────────────────────────────────
            print(f"\n  --- Phase 2b: Teacher training on pool K={pool.K} ---")
            teacher, teacher_eval = train_teacher(
                teacher    = teacher,
                pool       = pool,
                n_per_gen  = n_per_gen,
                device     = device,
                epochs     = teacher_epochs,
            )
            teacher_eval_log[t] = teacher_eval

            teacher_path = os.path.join(models_dir, f'teacher_t{t}_K{pool.K}.pt')
            torch.save(teacher.state_dict(), teacher_path)
            print(f"  [Server] Teacher saved → {teacher_path}")

            # ── Phase 4 ───────────────────────────────────────────────
            _run_phase4(client_list, teacher, pool, tag=f't{t}_K{pool.K}')

    # ── Save results ──────────────────────────────────────────────────
    print("\n=== EFL Complete ===")

    results = {
        'pool_size_per_step' : pool_size_log,
        'pool_labels'        : pool.labels,
        'mmd_admission_log'  : pool.mmd_log,
        'mmd_matrices'       : mmd_matrices,
        'generator_quality'  : generator_quality,
        'teacher_eval'       : teacher_eval_log,
        'student_eval'       : student_eval_log,
        'hyperparams': {
            'tau_merge'     : tau_merge,
            'fingerprint_P' : fingerprint_P,
            'n_per_gen'     : n_per_gen,
            'teacher_epochs': teacher_epochs,
        },
    }

    results_folder = os.path.join(project_root, 'results', attacktype)
    os.makedirs(results_folder, exist_ok=True)
    save_results_as_json('efl_metrics.json', results, results_folder, results_folder)
    return results
