# client_EFL.py
#
# Phase 1 (Algorithm 1):
#   1. Train ZI-RVAE generator on local (N, T, F) windows.
#   2. Evaluate generator quality with TSTR and TRTS using a small
#      LSTM trained on synthetic / tested on real (and vice versa).
#   3. Push only the decoder to the server (encoder stays on client).
#
# All other phases (Watcher, Distilled Replay Update, Student) are
# stubbed out and can be uncommented when those phases are enabled.

import gc
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import utils
from models import (ZIRVAEDecoder,
                    StudentModel,
                    TeacherModel,
                    TabDDPMGenerator,
                    detect_zi_feature_types, train_zirvae, train_tabddpm)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# ─────────────────────────────────────────────────────────────────────
# MMD² between real and synthetic (RBF kernel, median bandwidth)
# ─────────────────────────────────────────────────────────────────────

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    diff = X[:, None, :] - Y[None, :, :]
    return np.exp(-(diff ** 2).sum(axis=-1) / (2.0 * sigma ** 2))

def _mmd2_real_vs_syn(X_real: np.ndarray, X_syn: np.ndarray,
                      n_samples: int = 500) -> float:
    """
    Unbiased MMD² between real and synthetic windows.
    Both inputs are (N, T, F); flattened to (n_samples, T*F) for comparison.
    Sub-samples to n_samples for speed.
    """
    rng = np.random.default_rng(seed=0)
    def _subsample(X):
        idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
        return X[idx].reshape(len(idx), -1).astype(np.float32)

    X = _subsample(X_real)
    Y = _subsample(X_syn)
    P = min(len(X), len(Y))
    X, Y = X[:P], Y[:P]

    Z      = np.vstack([X, Y])
    diff   = Z[:, None, :] - Z[None, :, :]
    sq     = (diff ** 2).sum(axis=-1)
    sigma  = float(np.sqrt(np.median(sq[sq > 0]) / 2.0 + 1e-8))

    Kxx = _rbf_kernel(X, X, sigma); np.fill_diagonal(Kxx, 0.0)
    Kyy = _rbf_kernel(Y, Y, sigma); np.fill_diagonal(Kyy, 0.0)
    Kxy = _rbf_kernel(X, Y, sigma)

    return float(max(0.0,
        Kxx.sum() / (P * (P - 1)) +
        Kyy.sum() / (P * (P - 1)) -
        2.0 * Kxy.sum() / (P ** 2)
    ))


# ─────────────────────────────────────────────────────────────────────
# Tiny LSTM trained / evaluated for TSTR and TRTS
# ─────────────────────────────────────────────────────────────────────

def _train_eval_lstm(X_tr: np.ndarray, y_tr: np.ndarray,
                     X_te: np.ndarray, y_te: np.ndarray,
                     n_features: int, device: str,
                     hidden_dim: int = 64,
                     epochs: int = 30,
                     batch_size: int = 128,
                     lr: float = 1e-3) -> dict:
    """
    Train a StudentModel on (X_tr, y_tr) and test on (X_te, y_te).
    Uses the same architecture as the deployed client Student so that
    TSTR/TRTS scores reflect real task performance, not a proxy model.

    X_tr / X_te : (N, T, F)  float32 in [0, 1]
    y_tr / y_te : (N,)       int64   {0, 1}

    Returns dict with keys: accuracy, f1_macro, f1_weighted
    """
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long,    device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long,    device=device)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True,
    )

    clf = StudentModel(
        input_dim     = n_features,
        hidden_dim    = hidden_dim,
        num_layers    = 2,
        fc_hidden_dim = 32,
    ).to(device)

    opt = optim.Adam(clf.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()

    clf.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            ce(clf(xb), yb).backward()
            opt.step()

    clf.eval()
    with torch.no_grad():
        preds = torch.argmax(clf(X_te_t), dim=1).cpu().numpy()
    labels = y_te_t.cpu().numpy()

    return {
        'accuracy'    : float(accuracy_score(labels, preds)),
        'f1_macro'    : float(f1_score(labels, preds, average='macro',    zero_division=0)),
        'f1_weighted' : float(f1_score(labels, preds, average='weighted', zero_division=0)),
    }


class _ConditionalTabDDPM:
    """
    Wraps per-class TabDDPM models so the server pool can call
    generate(n, device) without knowing about the class structure.
    Generates equal samples from each class and concatenates.
    """
    def __init__(self, models: dict):
        self.models      = models   # {class_label: TabDDPMGenerator}
        # expose window_size / n_features / data_dim from first model
        first = next(iter(models.values()))
        self.window_size = first.window_size
        self.n_features  = first.n_features
        self.data_dim    = first.data_dim
        self.n_steps     = first.n_steps

    def generate(self, n: int, device: str = 'cpu') -> np.ndarray:
        n_per = max(1, n // len(self.models))
        parts = [m.generate(n=n_per, device=device)
                 for m in self.models.values()]
        return np.concatenate(parts, axis=0)

    def state_dict(self):
        return {str(k): m.state_dict() for k, m in self.models.items()}

    def load_state_dict(self, sd):
        for k, m in self.models.items():
            m.load_state_dict(sd[str(k)])

    def eval(self):
        for m in self.models.values():
            m.eval()


class ClientEFL:

    def __init__(self, client_id, args, domain_path,
                 assigned_domains, device,
                 tau_anom:       float = 0.5,
                 lambda2:        float = 1.0,
                 distil_tau:     float = 3.0,
                 watcher_window: int   = 50):

        self.client_id        = client_id
        self.args             = args
        self.domain_path      = domain_path
        self.assigned_domains = assigned_domains
        self.device           = device

        self.window_size    = args.window_size
        self.n_raw_features = args.n_raw_features

        # domain metadata only — data is loaded lazily in train_generator
        self._domains = utils.create_domains(domain_path, assigned_domains)
        self.domain_keys = list(self._domains.keys())

        # domain_key → {'model': ZIRVAE (frozen), 'tstr': dict, 'trts': dict}
        self.generators: dict = {}

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC — train_generator()   Phase 1 / Algorithm 1
    # ═══════════════════════════════════════════════════════════════
    def train_generator(self, time_step: int) -> tuple:
        """
        1. Collect (N, T, F) training windows from domain[time_step].
        2. Auto-detect feature types.
        3. Train ZI-RVAE.
        4. Evaluate with TSTR and TRTS.
        5. Freeze full model locally; return only the decoder to server.
        """
        domain   = self.domain_keys[time_step]
        gen_mode = getattr(self.args, 'generator', 'both')
        print(f"\n  [Client {self.client_id}] "
              f"Phase 1 — generator training [{gen_mode}] for '{domain}' …")

        # ── 1. Load this domain's data lazily (freed after training) ──
        files = self._domains[domain]
        train_loader, test_loader = utils.load_data(
            self.domain_path, domain, files,
            window_size    = self.window_size,
            step_size      = self.args.step_size,
            batch_size     = self.args.batch_size,
            n_raw_features = self.n_raw_features,
        )

        all_X, all_y = [], []
        for data, labels in train_loader:
            all_X.append(data.cpu().numpy())
            all_y.append(labels.cpu().numpy())
        X_real_tr = np.concatenate(all_X, axis=0)  # (N_tr, T, F)
        y_real_tr = np.concatenate(all_y, axis=0)
        del train_loader, all_X, all_y

        te_X, te_y = [], []
        for data, labels in test_loader:
            te_X.append(data.cpu().numpy())
            te_y.append(labels.cpu().numpy())
        X_real_te = np.concatenate(te_X, axis=0)   # (N_te, T, F)
        y_real_te = np.concatenate(te_y, axis=0)
        del test_loader, te_X, te_y

        N_tr, T, F = X_real_tr.shape
        print(f"  [Client {self.client_id}] "
              f"Real data — train: {X_real_tr.shape}  test: {X_real_te.shape}")

        # ── 2. Auto-detect feature distribution types ───────────────
        feat_type_idx = detect_zi_feature_types(X_real_tr)
        _counts = {k: len(v) for k, v in feat_type_idx.items() if v}
        print(f"  [Client {self.client_id}] Feature types: {_counts}")

        print(f"  [Client {self.client_id}] Generator mode: '{gen_mode}'  "
              f"(zirvae={'YES' if gen_mode in ('zirvae','both') else 'NO'}  "
              f"tabddpm={'YES' if gen_mode in ('tabddpm','both') else 'NO'})")

        n_syn    = min(N_tr, 1000)
        y_syn    = self._sample_labels_like(y_real_tr, n_syn)

        def _eval_gen(gen, name):
            X_syn = gen.generate(n=n_syn, device=str(self.device))
            tstr  = _train_eval_lstm(X_syn, y_syn, X_real_te, y_real_te, F, str(self.device))
            trts  = _train_eval_lstm(X_real_tr, y_real_tr, X_syn, y_syn, F, str(self.device))
            mmd2  = _mmd2_real_vs_syn(X_real_tr, X_syn)
            print(f"  [Client {self.client_id}] {name} — "
                  f"TSTR f1={tstr['f1_macro']:.4f}  "
                  f"TRTS f1={trts['f1_macro']:.4f}  "
                  f"MMD²={mmd2:.6f}")
            return tstr, trts, mmd2, X_syn

        decoder_zi   = None
        tabddpm      = None
        entry        = {}

        # ── 3a. Train ZI-RVAE (if requested) ──────────────────────
        if gen_mode in ('zirvae', 'both'):
            print(f"  [Client {self.client_id}] Training ZI-RVAE …")
            cfg = self._zirvae_config(F, N_tr)
            zirvae, recon_history = train_zirvae(
                X_np          = X_real_tr,
                window_size   = T,
                n_features    = F,
                feat_type_idx = feat_type_idx,
                hidden_dim    = cfg['hidden_dim'],
                latent_dim    = cfg['latent_dim'],
                n_layers      = cfg['n_layers'],
                epochs        = cfg['epochs'],
                batch_size    = cfg['batch_size'],
                lr            = cfg['lr'],
                noise_std     = cfg['noise_std'],
                free_bits     = cfg['free_bits'],
                n_cycles      = cfg['n_cycles'],
                device        = str(self.device),
                client_id     = self.client_id,
            )
            n_ep = len(recon_history)
            ckpts = [0, n_ep//4, n_ep//2, 3*n_ep//4, n_ep-1]
            ckpt_str = "  ".join(
                f"ep{ckpts[i]+1}:{recon_history[ckpts[i]]:.4f}"
                for i in range(len(ckpts)) if ckpts[i] < n_ep
            )
            print(f"  [Client {self.client_id}] ZI-RVAE recon — {ckpt_str}"
                  f"  final={recon_history[-1]:.4f}  best={min(recon_history):.4f}")

            tstr_zi, trts_zi, mmd2_zi, X_syn_zi = _eval_gen(zirvae, 'ZI-RVAE')
            decoder_zi = zirvae.decoder
            decoder_zi.eval()
            for p in decoder_zi.parameters():
                p.requires_grad_(False)
            del zirvae, X_syn_zi
            entry.update({
                'decoder'      : decoder_zi,
                'tstr'         : tstr_zi,
                'trts'         : trts_zi,
                'mmd2'         : mmd2_zi,
                'recon_history': recon_history,
            })

        # ── 3b. Train TabDDPM — one model per class ────────────────
        if gen_mode in ('tabddpm', 'both'):
            ddpm_cfg = self._tabddpm_config(N_tr)
            ddpm_models      = {}
            ddpm_loss_history = {}

            for cls_label, cls_name in [(0, 'benign'), (1, 'attack')]:
                idx = np.where(y_real_tr == cls_label)[0]
                if len(idx) < 10:
                    print(f"  [Client {self.client_id}] TabDDPM — "
                          f"skipping class {cls_name} (only {len(idx)} samples)")
                    continue
                X_cls = X_real_tr[idx]
                print(f"  [Client {self.client_id}] Training TabDDPM "
                      f"({cls_name}, n={len(X_cls)}) …")
                gen, loss_hist = train_tabddpm(
                    X_np        = X_cls,
                    window_size = T,
                    n_features  = F,
                    epochs      = ddpm_cfg['epochs'],
                    batch_size  = ddpm_cfg['batch_size'],
                    lr          = ddpm_cfg['lr'],
                    hidden_dim  = ddpm_cfg['hidden_dim'],
                    n_layers    = ddpm_cfg['n_layers'],
                    n_steps     = ddpm_cfg['n_steps'],
                    device      = str(self.device),
                    client_id   = self.client_id,
                )
                print(f"  [Client {self.client_id}] TabDDPM {cls_name} — "
                      f"final loss={loss_hist[-1]:.4f}  best={min(loss_hist):.4f}")
                ddpm_models[cls_label]       = gen
                ddpm_loss_history[cls_name]  = loss_hist

                # ── checkpoint immediately after training ──────────
                attacktype  = getattr(self.args, 'attacktype', None) or 'all'
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ckpt_dir    = os.path.join(project_root, 'saved_models_efl', attacktype)
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    ckpt_dir,
                    f"tabddpm_client{self.client_id}_{domain}_{cls_name}.pt"
                )
                torch.save({
                    'denoiser_state_dict': gen.denoiser.state_dict(),
                    'data_dim'           : gen.data_dim,
                    'window_size'        : gen.window_size,
                    'n_features'         : gen.n_features,
                    'n_steps'            : gen.n_steps,
                    'class_label'        : cls_label,
                    'class_name'         : cls_name,
                    'loss_history'       : loss_hist,
                }, ckpt_path)
                print(f"  [Client {self.client_id}] TabDDPM {cls_name} saved → {ckpt_path}")

            # build a combined generator wrapper for evaluation
            tabddpm = _ConditionalTabDDPM(ddpm_models)

            # evaluate with correct labels (deterministic per class)
            n_per_cls = n_syn // max(len(ddpm_models), 1)
            X_syn_parts, y_syn_parts = [], []
            for cls_label, gen in ddpm_models.items():
                X_k = gen.generate(n=n_per_cls, device=str(self.device))
                X_syn_parts.append(X_k)
                y_syn_parts.append(np.full(n_per_cls, cls_label, dtype=np.int64))
            X_syn_ddpm = np.concatenate(X_syn_parts)
            y_syn_ddpm = np.concatenate(y_syn_parts)

            tstr_ddpm = _train_eval_lstm(X_syn_ddpm, y_syn_ddpm,
                                         X_real_te,  y_real_te, F, str(self.device))
            trts_ddpm = _train_eval_lstm(X_real_tr,  y_real_tr,
                                         X_syn_ddpm, y_syn_ddpm, F, str(self.device))
            mmd2_ddpm = _mmd2_real_vs_syn(X_real_tr, X_syn_ddpm)
            print(f"  [Client {self.client_id}] TabDDPM (conditional) — "
                  f"TSTR f1={tstr_ddpm['f1_macro']:.4f}  "
                  f"TRTS f1={trts_ddpm['f1_macro']:.4f}  "
                  f"MMD²={mmd2_ddpm:.6f}")
            del X_syn_ddpm, X_syn_parts, y_syn_parts

            entry.update({
                'tabddpm'          : tabddpm,
                'tstr_ddpm'        : tstr_ddpm,
                'trts_ddpm'        : trts_ddpm,
                'mmd2_ddpm'        : mmd2_ddpm,
                'ddpm_loss_history': ddpm_loss_history,
            })

        # ── 4. Cleanup ─────────────────────────────────────────────
        del X_real_tr, y_real_tr, X_real_te, y_real_te, y_syn
        gc.collect()
        if str(self.device) == 'mps':
            torch.mps.empty_cache()
        elif str(self.device).startswith('cuda'):
            torch.cuda.empty_cache()

        self.generators[domain] = entry

        print(f"  [Client {self.client_id}] Generator(s) trained ✓  "
              f"total domains: {len(self.generators)}")
        print(f"  [Client {self.client_id}] Pushing generator(s) for '{domain}' to server.")
        return decoder_zi, tabddpm

    # ═══════════════════════════════════════════════════════════════
    # PRIVATE helpers
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _sample_labels_like(y_real: np.ndarray, n: int) -> np.ndarray:
        """
        Sample n labels preserving the class distribution of y_real.
        Used to assign labels to unconditionally generated synthetic windows.
        """
        classes, counts = np.unique(y_real, return_counts=True)
        probs   = counts / counts.sum()
        return np.random.choice(classes, size=n, p=probs).astype(np.int64)

    def _zirvae_config(self, n_features: int, n_samples: int) -> dict:
        hidden_dim = max(256, n_features * 16)
        latent_dim = max(32,  n_features)
        batch_size = min(256, max(32, n_samples // 20))
        epochs     = getattr(self.args, 'zirvae_epochs', 200)
        return {
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'n_layers'  : 2,
            'epochs'    : epochs,
            'free_bits' : 0.0,   # no floor — let KL find its natural level
            'n_cycles'  : 1,    # unused now (monotone schedule used instead)
            'noise_std' : 0.0,
            'batch_size': batch_size,
            'lr'        : 1e-3,
        }

    def _tabddpm_config(self, n_samples: int) -> dict:
        batch_size = min(256, max(32, n_samples // 20))
        epochs     = getattr(self.args, 'tabddpm_epochs', 500)
        return {
            'hidden_dim': 512,
            'n_layers'  : 4,
            'n_steps'   : 1000,
            'epochs'    : epochs,
            'batch_size': batch_size,
            'lr'        : 2e-4,
        }

    # ═══════════════════════════════════════════════════════════════
    # Baseline — train Student on real data only (no distillation)
    # ═══════════════════════════════════════════════════════════════

    def train_student_baseline(self, epochs: int = 30, lr: float = 1e-3):
        """
        Train a fresh Student using only real local data.
        Loss: L_task = CrossEntropy(Student(x_real), y_real)
        No Teacher, no synthetic replay.
        Result stored in self.student_baseline for later comparison.
        """
        device = self.device
        F = self.n_raw_features

        # fresh model — independent of any distilled student
        baseline = StudentModel(
            input_dim     = F,
            hidden_dim    = 64,
            num_layers    = 2,
            fc_hidden_dim = 32,
        ).to(device)

        # collect real training data from all domains
        all_X, all_y = [], []
        for domain in self.domain_keys:
            files = self._domains[domain]
            tr_loader, _ = utils.load_data(
                self.domain_path, domain, files,
                window_size    = self.window_size,
                step_size      = self.args.step_size,
                batch_size     = self.args.batch_size,
                n_raw_features = self.n_raw_features,
            )
            for xb, yb in tr_loader:
                all_X.append(xb.cpu().numpy())
                all_y.append(yb.cpu().numpy())

        X_real = np.concatenate(all_X, axis=0)
        y_real = np.concatenate(all_y, axis=0)
        del all_X, all_y

        X_t = torch.tensor(X_real, dtype=torch.float32)
        y_t = torch.tensor(y_real, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

        opt = optim.Adam(baseline.parameters(), lr=lr)
        ce  = nn.CrossEntropyLoss()
        baseline.train()

        bar = tqdm(range(epochs),
                   desc=f"Client {self.client_id} baseline",
                   leave=True, dynamic_ncols=True)
        for _ in bar:
            ep_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = ce(baseline(xb), yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            bar.set_postfix(loss=f"{ep_loss/max(len(loader),1):.4f}")

        baseline.eval()
        self.student_baseline = baseline
        print(f"  [Client {self.client_id}] Baseline Student trained "
              f"({epochs} epochs, real data only).")

    def evaluate_student_baseline(self) -> dict:
        """Evaluate self.student_baseline on test data from all trained domains."""
        if not hasattr(self, 'student_baseline') or self.student_baseline is None:
            print(f"  [Client {self.client_id}] No baseline student to evaluate.")
            return {}
        return self._evaluate_model(self.student_baseline, label="Baseline Student")

    # ═══════════════════════════════════════════════════════════════
    # Phase 4 — Distilled Replay
    # ═══════════════════════════════════════════════════════════════

    def receive_teacher(self, teacher: TeacherModel):
        """Store frozen Teacher broadcast from server."""
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher
        print(f"  [Client {self.client_id}] Teacher received "
              f"(K={teacher.head.out_features} classes).")

    def distilled_replay_update(self,
                                pool_decoders: list,
                                n_syn:         int   = 200,
                                epochs:        int   = 30,
                                lr:            float = 1e-3,
                                lambda2:       float = 1.0,
                                tau:           float = 3.0):
        """
        Phase 4 — train local Student with two losses:

          L_task   = CrossEntropy(Student(x_real), y_real)
          L_distil = τ² · KL( p_T(x_syn) ∥ p_S(x_syn) )
          L_total  = L_task + λ2 · L_distil

        The Teacher's K-class output is collapsed to 2-dim:
          p_attack = max(softmax(teacher_logits))
          p_benign = 1 - p_attack
        so it matches the Student's binary output.
        """
        if not hasattr(self, 'teacher') or self.teacher is None:
            print(f"  [Client {self.client_id}] No teacher — skipping distillation.")
            return

        device = str(self.device)

        # ── collect real data from all trained domains ─────────────
        all_X_real, all_y_real = [], []
        for domain in self.domain_keys:
            files = self._domains[domain]
            tr_loader, _ = utils.load_data(
                self.domain_path, domain, files,
                window_size    = self.window_size,
                step_size      = self.args.step_size,
                batch_size     = self.args.batch_size,
                n_raw_features = self.n_raw_features,
            )
            for xb, yb in tr_loader:
                all_X_real.append(xb.cpu().numpy())
                all_y_real.append(yb.cpu().numpy())
        X_real = np.concatenate(all_X_real, axis=0)   # (N, T, F)
        y_real = np.concatenate(all_y_real, axis=0)   # (N,)
        del all_X_real, all_y_real

        # ── generate synthetic replay from pool decoders ───────────
        all_X_syn = []
        for dec in pool_decoders:
            X_k = dec.generate(n=n_syn, device=device)  # (n_syn, T, F)
            all_X_syn.append(X_k)
        X_syn = np.concatenate(all_X_syn, axis=0).astype(np.float32)
        del all_X_syn

        # ── initialise Student if not yet created ──────────────────
        F = self.n_raw_features
        if not hasattr(self, 'student') or self.student is None:
            self.student = StudentModel(
                input_dim     = F,
                hidden_dim    = 64,
                num_layers    = 2,
                fc_hidden_dim = 32,
            ).to(self.device)

        self.student.train()
        self.teacher.to(self.device)

        opt = optim.Adam(self.student.parameters(), lr=lr)
        ce  = nn.CrossEntropyLoss()
        kl  = nn.KLDivLoss(reduction='batchmean')

        # tensors
        X_real_t = torch.tensor(X_real, dtype=torch.float32)
        y_real_t = torch.tensor(y_real, dtype=torch.long)
        X_syn_t  = torch.tensor(X_syn,  dtype=torch.float32)

        real_loader = DataLoader(
            TensorDataset(X_real_t, y_real_t),
            batch_size=64, shuffle=True)

        bar = tqdm(range(epochs),
                   desc=f"Client {self.client_id} distil",
                   leave=True, dynamic_ncols=True)

        for _ in bar:
            task_loss_sum = distil_loss_sum = 0.0

            # ── task loss on real data ─────────────────────────────
            for xb, yb in real_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits_s = self.student(xb)
                l_task   = ce(logits_s, yb)
                l_task.backward()
                opt.step()
                task_loss_sum += l_task.item()

            # ── distillation loss on synthetic data ────────────────
            # process in batches to avoid OOM
            n_syn_total = len(X_syn_t)
            for start in range(0, n_syn_total, 256):
                xb_syn = X_syn_t[start:start+256].to(self.device)

                with torch.no_grad():
                    t_logits  = self.teacher(xb_syn)           # (B, K)
                    t_soft    = torch.softmax(t_logits / tau, dim=1)
                    # collapse K → 2: p_attack = max over attack classes
                    p_attack  = t_soft.max(dim=1).values        # (B,)
                    p_benign  = 1.0 - p_attack
                    t_collapsed = torch.stack([p_benign, p_attack], dim=1)  # (B,2)

                opt.zero_grad()
                s_logits    = self.student(xb_syn)              # (B, 2)
                s_log_soft  = torch.log_softmax(s_logits / tau, dim=1)
                l_distil    = (tau ** 2) * kl(s_log_soft, t_collapsed)
                (lambda2 * l_distil).backward()
                opt.step()
                distil_loss_sum += l_distil.item()

            bar.set_postfix(
                task=f"{task_loss_sum/max(len(real_loader),1):.4f}",
                distil=f"{distil_loss_sum:.4f}")

        self.student.eval()
        print(f"  [Client {self.client_id}] Distillation done.")

    def evaluate_student(self) -> dict:
        """Evaluate self.student on test data from all trained domains."""
        if not hasattr(self, 'student') or self.student is None:
            print(f"  [Client {self.client_id}] No student to evaluate.")
            return {}
        return self._evaluate_model(self.student, label="Distilled Student")

    # ═══════════════════════════════════════════════════════════════
    # Shared evaluation helper
    # ═══════════════════════════════════════════════════════════════

    def _evaluate_model(self, model: nn.Module, label: str = "Student") -> dict:
        """
        Run model on test data from all trained domains.
        Returns accuracy, f1_macro, f1_weighted, confusion_matrix.
        Prints a formatted result table.
        """
        all_preds, all_labels = [], []
        for domain in self.domain_keys:
            files = self._domains[domain]
            _, te_loader = utils.load_data(
                self.domain_path, domain, files,
                window_size    = self.window_size,
                step_size      = self.args.step_size,
                batch_size     = self.args.batch_size,
                n_raw_features = self.n_raw_features,
            )
            for xb, yb in te_loader:
                xb = xb.to(self.device)
                with torch.no_grad():
                    logits = model(xb)
                    preds  = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(yb.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        acc    = float(accuracy_score(y_true, y_pred))
        f1_mac = float(f1_score(y_true, y_pred, average='macro',    zero_division=0))
        f1_wei = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        cm     = confusion_matrix(y_true, y_pred).tolist()
        report = str(classification_report(y_true, y_pred,
                                           target_names=['benign', 'attack'],
                                           zero_division=0))

        print(f"\n  {'─'*52}")
        print(f"  Client {self.client_id} — {label}")
        print(f"  {'─'*52}")
        print(f"  Accuracy   : {acc:.4f}")
        print(f"  F1 macro   : {f1_mac:.4f}")
        print(f"  F1 weighted: {f1_wei:.4f}")
        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        print(f"               benign   attack")
        if len(cm) == 2:
            print(f"  benign    {cm[0][0]:>8}  {cm[0][1]:>8}")
            print(f"  attack    {cm[1][0]:>8}  {cm[1][1]:>8}")
        print(f"\n  Classification report:")
        for line in report.split('\n'):
            print(f"    {line}")
        print(f"  {'─'*52}\n")

        return {
            'accuracy'        : acc,
            'f1_macro'        : f1_mac,
            'f1_weighted'     : f1_wei,
            'confusion_matrix': cm,
        }
