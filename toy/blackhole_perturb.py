# blackhole_perturb.py
#
# Perturbation-based synthetic data generation for Blackhole attacks.
#
# Core insight (from data analysis)
# ──────────────────────────────────
# 94% of Blackhole attack rows are statistically indistinguishable from normal.
# The only consistent signal is in rank and rank.1:
#
#   Normal  rank:   mean=464  std=46   (lag-1 autocorr = 0.96)
#   Attack  rank:   mean=409  std=55   (lag-1 autocorr = 0.98)
#   Normal  rank.1: mean=189  std=48   (lag-1 autocorr = 0.99)
#   Attack  rank.1: mean=147  std=44   (lag-1 autocorr = 0.99)
#   RF importance:  rank=35%, rank.1=36%  →  71% of all discriminative signal
#
# All other features (dior, dios, tots, diar, disr, diss) show KS < 0.12 and
# together contribute only 22% of RF importance.  disr/diss are all-zero in both
# classes (0% importance).
#
# Strategy
# ─────────
# Normal class:
#   Train a simple GRU-VAE on normal-class windows only (10 features, dead
#   features disr/diss/disr.1/diss.1 excluded from the model).
#   Sample N synthetic normal windows from it.
#
# Attack class  — NO generative model, pure perturbation:
#   Step 1  Fit AR(1) parameters to real attack rank sequences
#             φ     = measured lag-1 autocorrelation of attack rank
#             μ_atk = attack marginal mean
#             σ_inn = innovation std  = σ_atk * sqrt(1 - φ²)
#   Step 2  For each synthetic attack window:
#             a. Draw a synthetic normal window from the VAE (background)
#             b. Replace rank and rank.1 columns with AR(1) samples whose
#                marginal matches the real attack distribution
#             c. Leave ALL other features unchanged
#             d. Clip to [0, 1]  (normalised space)
#
# Evaluation
# ──────────
#   Standard: KS per feature (cls 0 and 1), TSTR, TRTS, Baseline F1
#   Rank-specific: rank-only KS, rank-conditional F1 (classify using rank alone)
#   Tail: tail KS, tail coverage  (windows where rank < 10th pct of normal)
#
# Run:
#   conda run -n vinnova python toy/blackhole_perturb.py
#
# Output:
#   toy/blackhole_perturb_results/<variant>/
#   toy/blackhole_perturb_results/summary.csv
#   toy/blackhole_perturb_results/summary.json

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.metrics import f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F_torch

warnings.filterwarnings('ignore')

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
from rvae import REncoder

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'attack_data')
OUT_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'blackhole_perturb_results')
os.makedirs(OUT_ROOT, exist_ok=True)

VARIANTS      = [d for d in sorted(os.listdir(DATA_ROOT)) if d.startswith('blackhole_')]
WINDOW_SIZE   = 10
N_SYNTH       = 1000          # synthetic windows per class
VAE_EPOCHS    = 300           # normal-class VAE only
DEVICE        = 'cpu'         # small model — CPU wins over MPS dispatch overhead
RNG_SEED      = 42
SPARSE_THRESH = 0.30
RANK_COLS     = ['rank', 'rank.1']

# These are always zero in both classes for Blackhole — excluded from VAE input
DEAD_FEATURES = ['disr', 'diss', 'disr.1', 'diss.1']


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (identical split logic to zirvae_multifile.py)
# ─────────────────────────────────────────────────────────────────────────────

def split_files(folder_path: str):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    rng       = np.random.default_rng(seed=RNG_SEED)
    shuffled  = rng.permutation(all_files).tolist()

    def _load(files):
        dfs = []
        for fname in files:
            df = pd.read_csv(os.path.join(folder_path, fname),
                             encoding='utf-8', encoding_errors='ignore')
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    df_tr = _load(shuffled[:14])
    df_te = _load(shuffled[14:])
    # Keep raw copies (before any preprocessing) for AR(1) fitting on rank levels
    df_tr_raw = df_tr.copy()
    df_te_raw = df_te.copy()
    return df_tr, df_te, df_tr_raw, df_te_raw


def preprocess_and_window(df_tr: pd.DataFrame, df_te: pd.DataFrame):
    """
    Preprocessing fitted on train — RANK COLUMNS ARE NOT DIFFERENCED.

    Blackhole's attack signal is in the absolute LEVEL of rank (attack rank ≈ 409
    vs normal rank ≈ 470).  Differencing converts this persistent level shift into
    a near-zero rate-of-change, destroying the only discriminative signal.

    Steps:
      1. log1p on sparse cols (zero% > SPARSE_THRESH) — NOT rank
      2. Min-max normalisation → [0, 1] on all features (including raw rank level)

    No differencing, no symmetric rescale.  Rank goes directly from raw integer
    (e.g. 470 or 409) → min-max normalised ∈ [0, 1].

    Returns:
      X_tr, y_tr, X_te, y_te  — windowed arrays  (N, T, F)
      feat_cols                — list of all feature names (14 total)
      active_cols              — feat_cols minus DEAD_FEATURES  (10 features for VAE)
      preproc                  — dict of fitted scalers
    """
    feat_cols    = [c for c in df_tr.columns if c != 'label']
    rank_present = [c for c in RANK_COLS if c in feat_cols]

    # 1. log1p on sparse cols (rank excluded — level must be preserved)
    sparse_cols = [c for c in feat_cols
                   if c not in rank_present and (df_tr[c] == 0).mean() > SPARSE_THRESH]
    for df in [df_tr, df_te]:
        if sparse_cols:
            df[sparse_cols] = np.log1p(df[sparse_cols])

    # 2. Min-max fitted on train
    g_min  = df_tr[feat_cols].min()
    g_max  = df_tr[feat_cols].max()
    denom  = (g_max - g_min).replace(0, 1)
    for df in [df_tr, df_te]:
        df[feat_cols] = ((df[feat_cols] - g_min) / denom).clip(0, 1).fillna(0)

    preproc = dict(g_min=g_min, g_max=g_max, denom=denom,
                   sparse_cols=sparse_cols, rank_present=rank_present)

    def _windows(df):
        vals   = df[feat_cols].values.astype(np.float32)
        labels = df['label'].values.astype(int)
        X, y   = [], []
        for i in range(len(vals) - WINDOW_SIZE):
            lbl = 1 if 1 in labels[i:i + WINDOW_SIZE] else 0
            X.append(vals[i:i + WINDOW_SIZE])
            y.append(lbl)
        return np.array(X, np.float32), np.array(y, np.int64)

    X_tr, y_tr = _windows(df_tr)
    X_te, y_te = _windows(df_te)

    active_cols = [c for c in feat_cols if c not in DEAD_FEATURES]
    return X_tr, y_tr, X_te, y_te, feat_cols, active_cols, preproc


# ─────────────────────────────────────────────────────────────────────────────
# AR(1) PARAMETER FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_ar1(seqs: np.ndarray) -> dict:
    """
    Fit AR(1) parameters to a 2-D array of sequences (N, T).

    Model:  x_t = phi * (x_{t-1} - mu) + mu + epsilon_t
            epsilon_t ~ N(0, sigma_inn^2)

    Parameters:
      mu        = marginal mean
      std       = marginal std
      phi       = average lag-1 autocorrelation across sequences
      sigma_inn = innovation std  = std * sqrt(1 - phi^2)
    """
    mu  = float(seqs.mean())
    std = float(seqs.std())

    phi_list = []
    for seq in seqs:
        if seq.std() > 1e-6:
            r = np.corrcoef(seq[:-1], seq[1:])[0, 1]
            if np.isfinite(r):
                phi_list.append(r)
    phi = float(np.mean(phi_list)) if phi_list else 0.0
    phi = np.clip(phi, -0.999, 0.999)

    sigma_inn = std * np.sqrt(max(1 - phi**2, 1e-6))
    return {'mu': mu, 'std': std, 'phi': phi, 'sigma_inn': sigma_inn}


def fit_rank_ar1_raw(df_tr_raw: pd.DataFrame) -> dict:
    """
    Fit AR(1) on RAW (unprocessed) rank values, separately for each class.

    Blackhole signal:
      Normal rank:  mean ≈ 464, std ≈ 46,  phi ≈ 0.96
      Attack rank:  mean ≈ 409, std ≈ 55,  phi ≈ 0.98

    The level difference (≈55 units) is the primary discriminative signal.
    Fitting on preprocessed (differenced) rank destroys this — after diff,
    both classes have near-zero mean (~0.004 difference).

    Returns:
      { col: { 'normal': {mu, std, phi, sigma_inn},
               'attack': {mu, std, phi, sigma_inn} } }
    """
    params = {}
    for col in RANK_COLS:
        if col not in df_tr_raw.columns:
            continue
        col_params = {}
        for cls, cls_name in [(0, 'normal'), (1, 'attack')]:
            vals = df_tr_raw.loc[df_tr_raw['label'] == cls, col].values.astype(np.float64)
            n_win = len(vals) // WINDOW_SIZE
            if n_win == 0:
                col_params[cls_name] = {
                    'mu': float(vals.mean()), 'std': max(float(vals.std()), 1.0),
                    'phi': 0.9, 'sigma_inn': max(float(vals.std()), 1.0) * 0.44}
                continue
            seqs = vals[:n_win * WINDOW_SIZE].reshape(n_win, WINDOW_SIZE)
            col_params[cls_name] = fit_ar1(seqs)
        params[col] = col_params
        pn = col_params['normal']
        pa = col_params['attack']
        tqdm.write(f"    RAW AR(1) {col}: "
                   f"normal  mu={pn['mu']:.1f} std={pn['std']:.1f} phi={pn['phi']:.4f} | "
                   f"attack  mu={pa['mu']:.1f} std={pa['std']:.1f} phi={pa['phi']:.4f}")
    return params


def raw_rank_seq_to_preprocessed(raw_seqs: np.ndarray, col: str,
                                  preproc: dict) -> np.ndarray:
    """
    Apply the rank preprocessing chain (NO differencing) to raw rank sequences.

    raw_seqs : (N, T)  — raw rank values (float from AR(1) sampler)
    Returns  : (N, T)  — min-max normalised values in [0, 1]

    Rank is a discrete integer (routing hop count).  The AR(1) sampler produces
    continuous floats, so we round to the nearest integer before normalising.
    This snaps synthetic values onto the same discrete grid as real data,
    reproducing the multimodal spike structure visible in rank histograms.

      normalised = clip((round(raw) - g_min) / denom, 0, 1)
    """
    g_min_col = float(preproc['g_min'][col])
    denom_col = float(preproc['denom'][col])
    out = np.clip((np.round(raw_seqs).astype(np.float64) - g_min_col) / denom_col,
                  0.0, 1.0).astype(np.float32)
    return out


def sample_ar1(params: dict, n_windows: int, window_size: int,
               rng: np.random.Generator) -> np.ndarray:
    """
    Sample n_windows independent AR(1) sequences of length window_size.

    Each sequence starts by drawing x_0 ~ N(mu, std) (from the marginal),
    then evolves as:  x_t = phi*(x_{t-1} - mu) + mu + N(0, sigma_inn^2)

    Returns array of shape (n_windows, window_size).
    """
    mu        = params['mu']
    phi       = params['phi']
    sigma_inn = params['sigma_inn']
    std       = params['std']

    out = np.empty((n_windows, window_size), dtype=np.float32)

    # Initialise each sequence from the marginal distribution
    out[:, 0] = rng.normal(mu, std, size=n_windows).astype(np.float32)

    for t in range(1, window_size):
        noise       = rng.normal(0, sigma_inn, size=n_windows).astype(np.float32)
        out[:, t]   = phi * (out[:, t-1] - mu) + mu + noise

    return out


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL-CLASS VAE  (trained on 10 active features only)
# ─────────────────────────────────────────────────────────────────────────────

def _kl_weight(epoch, n_epochs, n_cycles=4):
    cycle_len = max(n_epochs / n_cycles, 1)
    return min(1.0, ((epoch % cycle_len) / cycle_len) * 2.0)


class ZICVAE(nn.Module):
    """
    Zero-Inflated Conditional GRU-VAE.

    Extends NormalVAE with a class label y ∈ {0, 1} fed to both encoder
    and decoder via a small learned embedding (label_dim=4).  Everything
    else — ZI gate, mixed loss, free-bits KL — is identical.

    Encoder: GRU input at each timestep = [x_t ; embed(y)]
    Decoder: fc_h0 and fc_input conditioned on [z ; embed(y)]

    This lets the model learn separate conditional distributions
    P(features | normal) and P(features | attack) in a single model,
    so sampling with y=1 gives an attack-class background whose non-rank
    features (dior, dios, tots, diar, …) reflect the true attack
    conditional rather than the normal-class distribution.
    """
    LABEL_DIM = 4   # embedding size for the binary class label

    def __init__(self, n_features, window_size, hidden_dim, latent_dim,
                 zi_idx, n_layers=1):
        super().__init__()
        self.latent_dim  = latent_dim
        self.window_size = window_size
        self.zi_idx      = zi_idx
        self.n_features  = n_features
        self.n_layers    = n_layers
        self.hidden_dim  = hidden_dim

        ld = self.LABEL_DIM

        # Label embedding shared by encoder and decoder
        self.label_emb = nn.Embedding(2, ld)

        # Encoder: GRU reads (F + label_dim) per timestep
        self.enc_gru   = nn.GRU(n_features + ld, hidden_dim, n_layers,
                                batch_first=True)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_lv     = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.fc_lv.bias, -1.0)   # avoid posterior collapse

        # Decoder: conditioned on [z ; embed(y)]
        zld = latent_dim + ld
        self.fc_h0    = nn.Linear(zld, n_layers * hidden_dim)
        self.fc_input = nn.Linear(zld, n_features)
        self.gru      = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_features)
        self.fc_gate  = nn.Linear(hidden_dim, len(zi_idx)) if zi_idx else None

    # ── encoder ──────────────────────────────────────────────────────────────
    def _encode(self, x, y):
        """x: (B,T,F)  y: (B,) long  →  mu, lv"""
        e   = self.label_emb(y).unsqueeze(1).expand(-1, x.size(1), -1)  # (B,T,ld)
        _, h = self.enc_gru(torch.cat([x, e], dim=-1))
        h    = h[-1]                                                      # (B, hidden)
        mu   = self.fc_mu(h)
        lv   = self.fc_lv(h)
        return mu, lv

    # ── decoder ──────────────────────────────────────────────────────────────
    def _decode(self, z, y):
        """z: (B, latent_dim)  y: (B,) long"""
        B   = z.size(0)
        e   = self.label_emb(y)                                    # (B, ld)
        zy  = torch.cat([z, e], dim=-1)                            # (B, zld)
        h0  = self.fc_h0(zy).view(self.n_layers, B, self.hidden_dim)
        inp = self.fc_input(zy).unsqueeze(1).expand(-1, self.window_size, -1)
        gru_out, _ = self.gru(inp, h0)
        out        = torch.sigmoid(self.fc_out(gru_out))           # (B, T, F)
        gate       = self.fc_gate(gru_out) if self.fc_gate else None
        return out, gate

    def forward(self, x, y):
        mu, lv    = self._encode(x, y)
        z         = REncoder.reparameterize(mu, lv)
        out, gate = self._decode(z, y)
        return out, gate, mu, lv

    @torch.no_grad()
    def sample(self, n, y_val: int, device='cpu'):
        """Sample n windows for class y_val ∈ {0, 1}."""
        z    = torch.randn(n, self.latent_dim, device=device)
        y    = torch.full((n,), y_val, dtype=torch.long, device=device)
        out, gate = self._decode(z, y)
        result    = out.clone()

        if gate is not None:
            gate_prob = torch.sigmoid(gate)
            gate_bin  = torch.bernoulli(gate_prob).bool()
            for s_local, s_global in enumerate(self.zi_idx):
                result[:, :, s_global] = torch.where(
                    gate_bin[:, :, s_local],
                    out[:, :, s_global],
                    torch.zeros_like(out[:, :, s_global])
                )
        return result.cpu().numpy().astype(np.float32)


def _vae_loss(x, out, gate, mu, lv, zi_idx, lognorm_idx, rank_idx,
              kl_weight, free_bits):
    """
    Mixed loss:
      rank features   → MSE  (continuous after sign-log1p differencing)
      lognorm features → MSE on log1p-normalised values (already done in preproc,
                         so plain MSE in [0,1] space ≈ lognormal NLL in log-space)
      zi features     → gate BCE + MSE on non-zero values
    KL: free-bits with cyclical annealing weight.
    """
    recon   = torch.tensor(0.0, device=x.device)
    n_terms = 0

    # Rank and lognorm: MSE
    for f in rank_idx + lognorm_idx:
        recon   += F_torch.mse_loss(out[:, :, f], x[:, :, f])
        n_terms += 1

    # ZI gate + masked MSE
    if gate is not None:
        for s_local, s_global in enumerate(zi_idx):
            gate_lbl = (x[:, :, s_global] > 0).float()
            recon   += 0.5 * F_torch.binary_cross_entropy_with_logits(
                gate[:, :, s_local], gate_lbl)
            mask = gate_lbl.bool()
            if mask.any():
                recon += 0.5 * F_torch.mse_loss(
                    out[:, :, s_global][mask], x[:, :, s_global][mask])
            n_terms += 1

    recon_total = float(x.shape[1] * x.shape[2]) * recon / max(n_terms, 1)
    kl_elem     = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
    kl_loss     = torch.clamp(kl_elem.mean(0), min=free_bits).sum()
    return recon_total + kl_weight * kl_loss, recon.item() / max(n_terms, 1)


def train_zicvae(X_all, y_all, window_size, feat_cols_active,
                 hidden_dim=128, latent_dim=None, epochs=300,
                 batch_size=256, lr=1e-3, noise_std=0.05,
                 free_bits=0.5, n_cycles=4, device='cpu'):
    """
    Train ZICVAE on all windows (both classes) with class labels.

    The ZI gate and mixed loss are identical to the old NormalVAE.
    The only addition is that y is passed to encoder and decoder so
    the model learns separate conditional distributions for each class.

    X_all : (N, T, F_active) — all training windows, active features only
    y_all : (N,) int64       — class labels 0/1
    """
    F_  = len(feat_cols_active)
    if latent_dim is None:
        latent_dim = max(8, min((window_size * F_) // 8, 32))

    # Classify active features (using normal-class windows for ZI threshold)
    X_normal    = X_all[y_all == 0]
    rank_idx    = [i for i, c in enumerate(feat_cols_active) if c in RANK_COLS]
    zi_idx      = [i for i, c in enumerate(feat_cols_active)
                   if c not in RANK_COLS and
                   (X_normal[:, :, i] == 0).mean() > SPARSE_THRESH]
    lognorm_idx = [i for i in range(F_)
                   if i not in rank_idx and i not in zi_idx]

    tqdm.write(f"    ZICVAE feature buckets:")
    tqdm.write(f"      rank    : {[feat_cols_active[i] for i in rank_idx]}")
    tqdm.write(f"      lognorm : {[feat_cols_active[i] for i in lognorm_idx]}")
    tqdm.write(f"      zi      : {[feat_cols_active[i] for i in zi_idx]}")
    tqdm.write(f"      latent={latent_dim}  hidden={hidden_dim}  batch={batch_size}")
    tqdm.write(f"      training on {len(X_all)} windows "
               f"({(y_all==0).sum()} normal, {(y_all==1).sum()} attack)")

    X_t = torch.tensor(X_all,  device=device)
    y_t = torch.tensor(y_all,  dtype=torch.long, device=device)
    mdl = ZICVAE(F_, window_size, hidden_dim, latent_dim,
                 zi_idx).to(device)
    opt    = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=1e-5)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=batch_size, shuffle=True)

    mdl.train()
    pbar = tqdm(range(epochs), desc='      ZICVAE', unit='ep',
                dynamic_ncols=True, leave=False)
    for epoch in pbar:
        kl_w = _kl_weight(epoch, epochs, n_cycles)
        ep_loss = 0.0; n_b = 0
        for batch, labels in loader:
            opt.zero_grad()
            noisy = (batch + noise_std * torch.randn_like(batch)).clamp(0, 1)
            # preserve exact zeros on ZI features so the gate learns correctly
            for s_global in zi_idx:
                zero_mask = (batch[:, :, s_global] == 0)
                noisy[:, :, s_global][zero_mask] = 0.0
            out, gate, mu_z, lv = mdl(noisy, labels)
            loss, _ = _vae_loss(batch, out, gate, mu_z, lv,
                                 zi_idx, lognorm_idx, rank_idx,
                                 kl_w, free_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item(); n_b += 1
        pbar.set_postfix(loss=f'{ep_loss/max(n_b,1):.4f}', kl_w=f'{kl_w:.2f}')
    pbar.close()
    mdl.eval()
    return mdl


# ─────────────────────────────────────────────────────────────────────────────
# PERTURBATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_attack_windows(attack_background: np.ndarray,
                             raw_rank_params: dict,
                             preproc: dict,
                             feat_cols: list,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Create synthetic attack windows by:
      1. Starting from a CVAE attack-class background (y=1 samples).
         Non-rank features (dior, dios, tots, diar, …) are drawn from
         P(features | attack) learned by the ZICVAE — not copied from
         the normal class as before.
      2. Replacing rank and rank.1 columns with AR(1) sequences sampled in
         RAW rank space (mean≈409 for attack vs ≈464 for normal), then
         converted back to the preprocessed normalised space.
         This guarantees the rank level-shift that is the primary
         Blackhole discriminative signal.
      3. Clipping to [0, 1]

    attack_background : (N, T, F) — CVAE samples with y=1 (preprocessed)
    raw_rank_params   : output of fit_rank_ar1_raw()
    preproc           : output of preprocess_and_window()
    Returns           : (N, T, F) — synthetic attack windows (preprocessed)
    """
    N, T, F_ = attack_background.shape
    attack    = attack_background.copy()

    for col, p_dict in raw_rank_params.items():
        if col not in feat_cols:
            continue
        idx      = feat_cols.index(col)
        p_attack = p_dict['attack']

        # Sample AR(1) sequences in RAW rank space
        raw_seqs = sample_ar1(p_attack, N, T, rng)   # (N, T) — raw rank values

        # Convert to preprocessed normalised space
        preprocessed = raw_rank_seq_to_preprocessed(raw_seqs, col, preproc)
        attack[:, :, idx] = preprocessed

    return np.clip(attack, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ZIRVAE GENERATOR  (zirvae_multifile.py approach, adapted for Blackhole)
#
# Key differences from the ZICVAE+AR(1) approach above:
#   • Rank is DIFFERENCED (sign-log1p) — the original zirvae_multifile preprocessing
#   • Two separate per-class models trained independently (not one conditional model)
#   • Four-way feature-type detection: bernoulli / zi_lognorm / lognormal / continuous
#   • No AR(1) perturbation — the VAE is expected to learn rank on its own
#   • The Bernoulli head handles the all-zero dead features natively
# ─────────────────────────────────────────────────────────────────────────────

# ── Feature-type detection (identical to zirvae_multifile.py) ────────────────

_BERN_ZERO_THRESH = 0.90   # ≥90% zeros + ≤3 unique values → Bernoulli
_ZI_ZERO_THRESH   = 0.30   # ≥30% zeros (not Bernoulli) → ZI-lognorm
_LOGNORM_SKEW     = 1.0    # skew > 1.0 (non-rank) → lognormal


def _detect_feature_types_zirvae(df_raw: pd.DataFrame, feat_cols: list) -> dict:
    """
    Assign each feature a distribution type from raw (un-transformed) training data.

      'bernoulli'  — near-binary (≥90% zeros, ≤3 unique values)
                     Catches disr, diss, disr.1, diss.1 which are all-zero.
      'zi_lognorm' — zero-inflated continuous (≥30% zeros, not Bernoulli)
                     Catches diar, diar.1 (~85% zeros) and other sparse cols.
      'lognormal'  — right-skewed continuous (skew > 1.0, non-negative, not ZI)
      'continuous' — everything else, including rank cols (signed after differencing)

    Returns dict { col_name: type_string }
    """
    from scipy.stats import skew as _skew
    types = {}
    for col in feat_cols:
        if col in RANK_COLS:
            types[col] = 'continuous'   # rank always continuous (signed diff)
            continue
        x        = df_raw[col].dropna().values
        n_unique = len(np.unique(x))
        zero_pct = float((x == 0).mean())
        col_skew = float(_skew(x))

        if zero_pct >= _BERN_ZERO_THRESH and n_unique <= 3:
            types[col] = 'bernoulli'
        elif zero_pct >= _ZI_ZERO_THRESH:
            types[col] = 'zi_lognorm'
        elif x.min() >= 0 and col_skew > _LOGNORM_SKEW:
            types[col] = 'lognormal'
        else:
            types[col] = 'continuous'
    return types


def _preprocess_and_window_zirvae(df_tr: pd.DataFrame,
                                   df_te: pd.DataFrame) -> tuple:
    """
    Preprocessing matching zirvae_multifile.py exactly:
      1. sign-log1p differencing on rank columns  (preserves zero-crossings,
         maps 0 → 0.5 after symmetric rescaling)
      2. log1p on sparse columns (zero% > SPARSE_THRESH)
      3. Min-max normalisation → [0, 1]
      4. Symmetric rescaling for rank: 0 → 0.5

    NOTE: This preprocessing DESTROYS the rank level-shift (see EXPLAINER).
    The ZIRVAE therefore cannot rely on the rank level to discriminate classes —
    it must learn the temporal pattern of rank changes instead.

    Returns:
      X_tr, y_tr, X_te, y_te  — windowed arrays  (N, T, F)
      feat_cols                — list of all feature names
      feat_type_idx            — { type_str: [feature_indices] }
      preproc_z                — dict of fitted scalers (for reference)
    """
    feat_cols        = [c for c in df_tr.columns if c != 'label']
    rank_present     = [c for c in RANK_COLS if c in feat_cols]

    # 1. Differencing + sign-log1p on rank cols
    for df in [df_tr, df_te]:
        if rank_present:
            diff = df[rank_present].diff().fillna(0)
            df[rank_present] = np.sign(diff) * np.log1p(np.abs(diff))

    # 2. log1p on sparse cols
    sparse_cols = [c for c in feat_cols
                   if (df_tr[c] == 0).mean() > SPARSE_THRESH]
    for df in [df_tr, df_te]:
        if sparse_cols:
            df[sparse_cols] = np.log1p(df[sparse_cols])

    # 3. Min-max fitted on train
    g_min  = df_tr[feat_cols].min()
    g_max  = df_tr[feat_cols].max()
    denom  = (g_max - g_min).replace(0, 1)
    for df in [df_tr, df_te]:
        df[feat_cols] = ((df[feat_cols] - g_min) / denom).clip(0, 1).fillna(0)

    # 4. Symmetric rescaling: 0 → 0.5 for rank (so the zero spike is centred)
    for col in rank_present:
        abs_max = float(df_tr[col].abs().max())
        if abs_max > 0:
            for df in [df_tr, df_te]:
                df[col] = (df[col] / (2 * abs_max) + 0.5).clip(0, 1)

    preproc_z = dict(g_min=g_min, g_max=g_max, denom=denom,
                     sparse_cols=sparse_cols, rank_present=rank_present)

    def _windows(df):
        vals   = df[feat_cols].values.astype(np.float32)
        labels = df['label'].values.astype(int)
        X, y   = [], []
        for i in range(len(vals) - WINDOW_SIZE):
            lbl = 1 if 1 in labels[i:i + WINDOW_SIZE] else 0
            X.append(vals[i:i + WINDOW_SIZE])
            y.append(lbl)
        return np.array(X, np.float32), np.array(y, np.int64)

    X_tr_w, y_tr_w = _windows(df_tr)
    X_te_w, y_te_w = _windows(df_te)

    return X_tr_w, y_tr_w, X_te_w, y_te_w, feat_cols, preproc_z


# ── ZIRVAE model (verbatim from zirvae_multifile.py) ─────────────────────────

class _ZIRVAEDecoder(nn.Module):
    """
    GRU decoder with per-feature-type output heads.

    Feature routing:
      bernoulli  → raw logit  (no sigmoid; BCE-with-logits in loss, sigmoid at sample time)
      zi_lognorm → sigmoid output + separate gate logit
      lognormal  → sigmoid output + MSE
      continuous → sigmoid output + MSE
    """
    def __init__(self, latent_dim, hidden_dim, n_features,
                 window_size, feat_type_idx, n_layers=1):
        super().__init__()
        self.window_size   = window_size
        self.n_layers      = n_layers
        self.hidden_dim    = hidden_dim
        self.feat_type_idx = feat_type_idx
        self.n_features    = n_features

        self.fc_h0    = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_input = nn.Linear(latent_dim, n_features)
        self.gru      = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_features)

        zi_idx       = feat_type_idx.get('zi_lognorm', [])
        self.gate_fc = nn.Linear(hidden_dim, len(zi_idx)) if zi_idx else None

    def forward(self, z):
        B          = z.size(0)
        h0         = self.fc_h0(z).view(self.n_layers, B, self.hidden_dim)
        inp        = self.fc_input(z).unsqueeze(1).expand(-1, self.window_size, -1)
        gru_out, _ = self.gru(inp, h0)
        raw_out    = self.fc_out(gru_out)          # (B, T, F)

        out = torch.sigmoid(raw_out).clone()

        # Bernoulli features keep raw logits (loss uses BCE-with-logits)
        for f in self.feat_type_idx.get('bernoulli', []):
            out[:, :, f] = raw_out[:, :, f].clone()

        gate_logit = self.gate_fc(gru_out) if self.gate_fc is not None else None
        return out, gate_logit


class _ZIRVAE(nn.Module):
    """
    Zero-Inflated RVAE (per-class, unconditional).

    Unlike ZICVAE, this model is trained on ONE class at a time.
    The encoder is a standard GRU-based RVAE encoder from rvae.py.
    The decoder is _ZIRVAEDecoder with four-way feature-type routing.
    """
    def __init__(self, n_features, window_size, hidden_dim,
                 latent_dim, feat_type_idx, n_layers=1):
        super().__init__()
        self.latent_dim    = latent_dim
        self.feat_type_idx = feat_type_idx
        self.encoder       = REncoder(n_features, hidden_dim, latent_dim, n_layers)
        self.decoder       = _ZIRVAEDecoder(latent_dim, hidden_dim, n_features,
                                            window_size, feat_type_idx, n_layers)

    def forward(self, x):
        mu, lv          = self.encoder(x)
        z               = REncoder.reparameterize(mu, lv)
        out, gate_logit = self.decoder(z)
        return out, gate_logit, mu, lv

    @torch.no_grad()
    def sample(self, n, device='cpu'):
        z               = torch.randn(n, self.latent_dim, device=device)
        out, gate_logit = self.decoder(z)
        result          = out.clone()

        for f in self.feat_type_idx.get('bernoulli', []):
            result[:, :, f] = torch.bernoulli(torch.sigmoid(out[:, :, f]))

        zi_idx = self.feat_type_idx.get('zi_lognorm', [])
        if gate_logit is not None and zi_idx:
            gate_prob = torch.sigmoid(gate_logit)
            gate      = torch.bernoulli(gate_prob).bool()
            for s_local, s_global in enumerate(zi_idx):
                result[:, :, s_global] = torch.where(
                    gate[:, :, s_local],
                    out[:, :, s_global],
                    torch.zeros_like(out[:, :, s_global])
                )
        return result.cpu().numpy().astype(np.float32)


def _zirvae_loss(x, out, gate_logit, mu, lv,
                 feat_type_idx, kl_weight, loss_factor, free_bits):
    """
    Mixed reconstruction loss (identical to zirvae_multifile.py):
      bernoulli  → BCE-with-logits  (out holds raw logit)
      zi_lognorm → 0.5 * gate BCE  +  0.5 * MSE on non-zero values
      lognormal  → MSE
      continuous → MSE
    KL: free-bits per dimension, scaled by kl_weight.
    """
    recon   = torch.tensor(0.0, device=x.device)
    n_terms = 0

    for f in feat_type_idx.get('bernoulli', []):
        target  = (x[:, :, f] > 0).float()
        recon  += F_torch.binary_cross_entropy_with_logits(
            out[:, :, f], target, reduction='mean')
        n_terms += 1

    zi_idx = feat_type_idx.get('zi_lognorm', [])
    if gate_logit is not None and zi_idx:
        for s_local, s_global in enumerate(zi_idx):
            gate_lbl = (x[:, :, s_global] > 0).float()
            recon   += 0.5 * F_torch.binary_cross_entropy_with_logits(
                gate_logit[:, :, s_local], gate_lbl, reduction='mean')
            mask = gate_lbl.bool()
            if mask.any():
                recon += 0.5 * F_torch.mse_loss(
                    out[:, :, s_global][mask], x[:, :, s_global][mask],
                    reduction='mean')
            n_terms += 1

    for type_key in ('lognormal', 'continuous'):
        for f in feat_type_idx.get(type_key, []):
            recon   += F_torch.mse_loss(out[:, :, f], x[:, :, f], reduction='mean')
            n_terms += 1

    recon_total = loss_factor * recon / max(n_terms, 1)
    kl_elem     = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
    kl_loss     = torch.clamp(kl_elem.mean(0), min=free_bits).sum()
    return recon_total + kl_weight * kl_loss, (recon / max(n_terms, 1)).item()


def train_zirvae_blackhole(X_cls: np.ndarray,
                           window_size: int,
                           feat_type_idx: dict,
                           hidden_dim: int = 128,
                           latent_dim: int = 16,
                           epochs: int = 300,
                           batch_size: int = 256,
                           lr: float = 1e-3,
                           noise_std: float = 0.05,
                           free_bits: float = 0.5,
                           n_cycles: int = 4,
                           device: str = 'cpu') -> _ZIRVAE:
    """
    Train a per-class ZIRVAE on Blackhole data.

    X_cls   : (N, T, F) windows for ONE class (normal or attack)
    Returns : trained _ZIRVAE model

    This is the same training loop as zirvae_multifile.py:train_zirvae(),
    adapted to accept (N, T, F) input directly (not flattened).
    """
    N, T, F_ = X_cls.shape
    X_t      = torch.tensor(X_cls, device=device)
    bern_idx = feat_type_idx.get('bernoulli', [])

    model  = _ZIRVAE(F_, T, hidden_dim, latent_dim,
                     feat_type_idx, n_layers=1).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True)
    loss_factor = float(T * F_)

    model.train()
    pbar = tqdm(range(epochs), desc='      ZIRVAE', unit='ep',
                dynamic_ncols=True, leave=False)
    for epoch in pbar:
        # Cyclical β annealing (same schedule as ZICVAE)
        cycle_len = max(epochs / n_cycles, 1)
        kl_w      = min(1.0, ((epoch % cycle_len) / cycle_len) * 2.0)

        ep_loss = 0.0; n_b = 0
        for (batch,) in loader:
            opt.zero_grad()
            noisy = (batch + noise_std * torch.randn_like(batch)).clamp(0, 1)
            # No noise on Bernoulli features (binary — noise corrupts the gate target)
            if bern_idx:
                noisy[:, :, bern_idx] = batch[:, :, bern_idx]
            out, gate_logit, mu_z, lv = model(noisy)
            loss, _ = _zirvae_loss(batch, out, gate_logit, mu_z, lv,
                                    feat_type_idx, kl_w, loss_factor, free_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item(); n_b += 1
        pbar.set_postfix(loss=f'{ep_loss/max(n_b,1):.4f}', kl_w=f'{kl_w:.2f}')
    pbar.close()
    model.eval()
    return model


def generate_zirvae(variant: str,
                    df_tr_raw: pd.DataFrame,
                    df_te_raw: pd.DataFrame,
                    n_synth: int = N_SYNTH,
                    epochs: int = VAE_EPOCHS,
                    device: str = DEVICE) -> tuple:
    """
    Generate synthetic Blackhole data using the zirvae_multifile.py approach:
      — separate per-class ZIRVAE (no conditioning)
      — rank columns are differenced (sign-log1p), NOT level-preserved
      — four-way feature-type detection (bernoulli / zi_lognorm / lognormal / continuous)
      — no AR(1) perturbation

    Parameters
    ----------
    variant     : variant name string (for logging only)
    df_tr_raw   : raw training DataFrame (before any preprocessing)
    df_te_raw   : raw test DataFrame (before any preprocessing)
    n_synth     : number of synthetic windows per class
    epochs      : VAE training epochs per class
    device      : torch device string

    Returns
    -------
    X_syn       : (2*n_synth, T, F) synthetic windows
    y_syn       : (2*n_synth,)      synthetic labels
    X_tr_z      : (N, T, F)         real training windows (ZIRVAE preprocessing)
    y_tr_z      : (N,)              real training labels
    X_te_z      : (N, T, F)         real test windows (ZIRVAE preprocessing)
    y_te_z      : (N,)              real test labels
    feat_cols_z : list of feature names
    feat_type_idx : { type_str: [feature indices] }
    """
    tqdm.write(f"\n  [ZIRVAE] Preprocessing with rank differencing …")

    # Detect feature types on raw data BEFORE any transformation
    feat_cols_raw = [c for c in df_tr_raw.columns if c != 'label']
    feat_types    = _detect_feature_types_zirvae(df_tr_raw, feat_cols_raw)
    tqdm.write(f"    Feature types detected:")
    for ftype in ('bernoulli', 'zi_lognorm', 'lognormal', 'continuous'):
        cols = [c for c, t in feat_types.items() if t == ftype]
        if cols:
            tqdm.write(f"      {ftype:12s}: {cols}")

    (X_tr_z, y_tr_z,
     X_te_z, y_te_z,
     feat_cols_z,
     _) = _preprocess_and_window_zirvae(df_tr_raw.copy(), df_te_raw.copy())

    T, F_ = X_tr_z.shape[1], X_tr_z.shape[2]

    # Build feat_type_idx with post-preprocessing feature order
    feat_type_idx: dict = {'bernoulli': [], 'zi_lognorm': [], 'lognormal': [], 'continuous': []}
    for col in feat_cols_z:
        ftype = feat_types.get(col, 'continuous')
        feat_type_idx[ftype].append(feat_cols_z.index(col))

    # Hyperparams (same auto-scaling as run_variant)
    latent_dim = max(8,  min((T * F_) // 8, 32))
    hidden_dim = max(64, F_ * 6)
    batch_size = min(512, max(64, len(X_tr_z) // 10))
    free_bits  = round(min(0.5, 8.0 / latent_dim), 4)
    tqdm.write(f"    latent={latent_dim}  hidden={hidden_dim}  "
               f"batch={batch_size}  free_bits={free_bits}")

    syn_parts_X, syn_parts_y = [], []

    for cls in [0, 1]:
        cls_label = 'Normal' if cls == 0 else 'Attack'
        X_cls     = X_tr_z[y_tr_z == cls]
        tqdm.write(f"\n  [ZIRVAE] Training class {cls} ({cls_label}) "
                   f"— {len(X_cls)} windows, {epochs} epochs …")

        if len(X_cls) < batch_size:
            tqdm.write(f"    [skip] too few samples ({len(X_cls)} < {batch_size})")
            continue

        model = train_zirvae_blackhole(
            X_cls      = X_cls,
            window_size= T,
            feat_type_idx = feat_type_idx,
            hidden_dim = hidden_dim,
            latent_dim = latent_dim,
            epochs     = epochs,
            batch_size = batch_size,
            noise_std  = 0.05,
            free_bits  = free_bits,
            device     = device,
        )
        X_syn_cls = model.sample(n_synth, device=device)   # (n_synth, T, F)
        syn_parts_X.append(X_syn_cls)
        syn_parts_y.append(np.full(n_synth, cls, dtype=np.int64))

    X_syn = np.concatenate(syn_parts_X, axis=0)
    y_syn = np.concatenate(syn_parts_y, axis=0)

    return (X_syn, y_syn,
            X_tr_z, y_tr_z, X_te_z, y_te_z,
            feat_cols_z, feat_type_idx)


def run_variant_zirvae(variant: str) -> dict:
    """
    Run the full per-variant pipeline using the ZIRVAE generator
    (zirvae_multifile.py approach) instead of ZICVAE + AR(1).

    Uses the same evaluation suite as run_variant() so results are
    directly comparable: KS per class, TSTR, TRTS, baseline F1,
    rank-only F1, tail KS, cond. precision.

    Output is written to:
      toy/blackhole_perturb_results/<variant>/zirvae_metrics.json
      toy/blackhole_perturb_results/<variant>/zirvae_distributions_*.png
      toy/blackhole_perturb_results/<variant>/zirvae_ks_*.png
    """
    tqdm.write(f"\n{'='*65}")
    tqdm.write(f"  VARIANT (ZIRVAE): {variant}")
    tqdm.write(f"{'='*65}")

    folder_path = os.path.join(DATA_ROOT, variant)
    out_dir     = os.path.join(OUT_ROOT, variant)
    os.makedirs(out_dir, exist_ok=True)

    # Load raw data (split_files gives us raw copies)
    df_tr, df_te, df_tr_raw, df_te_raw = split_files(folder_path)

    # Also run the standard preprocessing so we can compute rank-only F1
    # and baseline on the same windows as run_variant()
    (X_tr_std, y_tr_std,
     X_te_std, y_te_std,
     feat_cols_std, _, _) = preprocess_and_window(df_tr.copy(), df_te.copy())

    tqdm.write(f"  Train (std preproc): {X_tr_std.shape}  "
               f"Test: {X_te_std.shape}")

    # ── Baseline LSTM (standard preprocessing, all features) ─────────────────
    tqdm.write(f"\n  [1/3] Baseline LSTM (standard preproc) …")
    base_clf = train_lstm(X_tr_std, y_tr_std, device=DEVICE)
    base_f1  = eval_clf(base_clf, X_te_std, y_te_std, device=DEVICE)
    tqdm.write(f"       Baseline F1 = {base_f1:.4f}")

    rank_f1 = rank_only_f1(X_tr_std, y_tr_std, X_te_std, y_te_std,
                            feat_cols_std, device=DEVICE)
    tqdm.write(f"       Rank-only F1 = {rank_f1:.4f}")

    # ── ZIRVAE generation ─────────────────────────────────────────────────────
    tqdm.write(f"\n  [2/3] ZIRVAE generation ({VAE_EPOCHS} epochs per class) …")
    (X_syn, y_syn,
     X_tr_z, y_tr_z,
     X_te_z, y_te_z,
     feat_cols_z,
     feat_type_idx) = generate_zirvae(
        variant   = variant,
        df_tr_raw = df_tr_raw,
        df_te_raw = df_te_raw,
        n_synth   = N_SYNTH,
        epochs    = VAE_EPOCHS,
        device    = DEVICE,
    )

    # ── Evaluate on ZIRVAE-preprocessed windows ───────────────────────────────
    tqdm.write(f"\n  [3/3] Evaluating …")

    ks_res  = ks_per_class(X_tr_z, y_tr_z, X_syn, y_syn, feat_cols_z)
    tstr_f1 = eval_clf(train_lstm(X_syn,    y_syn,    device=DEVICE),
                       X_te_z,  y_te_z,  device=DEVICE)
    trts_f1 = eval_clf(train_lstm(X_tr_z,   y_tr_z,   device=DEVICE),
                       X_syn,   y_syn,   device=DEVICE)

    tail_ks_df, tail_cov_df = tail_ks_and_coverage(
        X_tr_z, y_tr_z, X_syn, y_syn, feat_cols_z,
        rank_col='rank', quantile=0.10)

    # Conditional precision: real clf on ZIRVAE-preprocessed space
    clf_real  = train_lstm(X_tr_z, y_tr_z, device=DEVICE)
    X_syn_atk = X_syn[y_syn == 1]
    Xt_atk    = torch.tensor(X_syn_atk, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        preds_atk = clf_real(Xt_atk).argmax(1).cpu().numpy()
    cond_precision = float((preds_atk == 1).mean()) if len(preds_atk) > 0 else 0.0

    for cls in [0, 1]:
        kdf   = ks_res[cls]
        label = 'Normal' if cls == 0 else 'Attack'
        tqdm.write(f"  KS cls{cls} ({label}): "
                   f"pass={kdf['similar'].sum()}/{len(kdf)}  "
                   f"mean_KS={kdf['ks_stat'].mean():.4f}")
    tqdm.write(f"  TSTR={tstr_f1:.4f}  TRTS={trts_f1:.4f}  "
               f"Baseline={base_f1:.4f}  RankOnly={rank_f1:.4f}")
    if not tail_ks_df.empty:
        tqdm.write(f"  Mean Tail KS (low rank): "
                   f"{tail_ks_df['tail_ks'].mean():.4f}")
    tqdm.write(f"  Cond. Precision: {cond_precision:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # Reuse plot helpers from run_variant(), with 'zirvae_' prefix on filenames
    _F = len(feat_cols_z)
    fig, axes = plt.subplots(2, _F, figsize=(2.8 * _F, 7), constrained_layout=True)
    for row, (cls, label) in enumerate([(0, 'Normal'), (1, 'Attack')]):
        r_flat = X_tr_z[y_tr_z == cls].mean(axis=1)
        s_flat = X_syn[y_syn == cls].mean(axis=1)
        for col_idx, col in enumerate(feat_cols_z):
            ax = axes[row, col_idx]
            r  = r_flat[:, col_idx]; s = s_flat[:, col_idx]
            if r.std() < 1e-8 and s.std() < 1e-8:
                ax.set_title(f'{col}\n(constant)', fontsize=6); continue
            try:
                ax.hist(r, bins=30, alpha=0.4, color='steelblue', density=True, label='Real')
                ax.hist(s, bins=30, alpha=0.4, color='tomato',    density=True, label='Syn')
            except Exception: pass
            try:
                xs = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 120)
                ax.plot(xs, scipy_stats.gaussian_kde(r)(xs), 'steelblue', lw=1.2)
                ax.plot(xs, scipy_stats.gaussian_kde(s)(xs), 'tomato',    lw=1.2)
            except Exception: pass
            ks_v, _ = scipy_stats.ks_2samp(r, s)
            ax.set_title(f'{col}\nKS={ks_v:.3f}', fontsize=6)
            ax.tick_params(labelsize=5)
            if col_idx == 0: ax.set_ylabel(label, fontsize=7)
            if row == 0 and col_idx == 0: ax.legend(fontsize=5)
    fig.suptitle(f'ZIRVAE — All feature distributions — {variant}', fontsize=9)
    fig.savefig(os.path.join(out_dir, f'zirvae_distributions_{variant}.png'),
                dpi=120, bbox_inches='tight')
    plt.close(fig)

    plot_ks_bars({0: ks_res[0], 1: ks_res[1]},
                 f'zirvae_{variant}', out_dir)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        'variant':             variant,
        'generator':           'ZIRVAE',
        'baseline_f1':         round(base_f1,  4),
        'rank_only_f1':        round(rank_f1,  4),
        'tstr_f1':             round(tstr_f1,  4),
        'trts_f1':             round(trts_f1,  4),
        'cond_precision':      round(cond_precision, 4),
        'ks_cls0_mean':        round(float(ks_res[0]['ks_stat'].mean()), 4),
        'ks_cls1_mean':        round(float(ks_res[1]['ks_stat'].mean()), 4),
        'ks_cls0_pass':        int(ks_res[0]['similar'].sum()),
        'ks_cls1_pass':        int(ks_res[1]['similar'].sum()),
        'mean_tail_ks':        round(float(tail_ks_df['tail_ks'].mean()), 4)
                               if not tail_ks_df.empty else None,
        'mean_tail_coverage':  round(float(tail_cov_df['tail_coverage'].mean()), 4)
                               if not tail_cov_df.empty else None,
        'feat_types':          {ftype: [feat_cols_z[i] for i in idxs]
                                for ftype, idxs in feat_type_idx.items() if idxs},
        'ks_per_feature_cls0': ks_res[0].to_dict(orient='records'),
        'ks_per_feature_cls1': ks_res[1].to_dict(orient='records'),
        'tail_ks':             tail_ks_df.to_dict(orient='records')
                               if not tail_ks_df.empty else [],
        'tail_coverage':       tail_cov_df.to_dict(orient='records')
                               if not tail_cov_df.empty else [],
    }
    with open(os.path.join(out_dir, 'zirvae_metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


def main_zirvae():
    """
    Entry point for running the ZIRVAE generator on all Blackhole variants.

    Produces a side-by-side summary with the perturbation approach so the two
    methods can be directly compared on the same variants and evaluation suite.

    Run:
      conda run -n vinnova python toy/blackhole_perturb.py --zirvae
    """
    print(f"ZIRVAE Blackhole synthesis — {len(VARIANTS)} variants")
    all_metrics = []

    outer = tqdm(VARIANTS, desc='Overall [ZIRVAE]', unit='variant',
                 dynamic_ncols=True, position=0)
    for variant in outer:
        outer.set_description(f'Overall [ZIRVAE] [{variant}]')
        try:
            m = run_variant_zirvae(variant)
            all_metrics.append(m)
            outer.set_postfix(
                tstr=f"{m['tstr_f1']:.3f}",
                ks1=f"{m['ks_cls1_mean']:.3f}")
        except Exception as e:
            import traceback
            tqdm.write(f"  ERROR in {variant}: {e}")
            traceback.print_exc()

    if not all_metrics:
        print("No variants processed.")
        return

    zirvae_out = os.path.join(OUT_ROOT, 'zirvae_summary.json')
    with open(zirvae_out, 'w') as fh:
        json.dump(all_metrics, fh, indent=2)

    flat = [{k: v for k, v in m.items() if not isinstance(v, (list, dict))}
            for m in all_metrics]
    pd.DataFrame(flat).to_csv(
        os.path.join(OUT_ROOT, 'zirvae_summary.csv'), index=False)

    print('\n' + '='*80)
    print('  FINAL SUMMARY — ZIRVAE Blackhole Synthesis')
    print('='*80)
    hdr = (f"  {'Variant':<28} {'Base':>6} {'RnkOnly':>8} "
           f"{'TSTR':>6} {'TRTS':>6} {'KS0':>6} {'KS1':>6} {'CondPrec':>9}")
    print(hdr); print('-' * len(hdr))
    for m in all_metrics:
        print(f"  {m['variant']:<28} {m['baseline_f1']:>6.3f} "
              f"{m['rank_only_f1']:>8.3f} "
              f"{m['tstr_f1']:>6.3f} {m['trts_f1']:>6.3f} "
              f"{m['ks_cls0_mean']:>6.3f} {m['ks_cls1_mean']:>6.3f} "
              f"{m['cond_precision']:>9.3f}")
    print(f"\nAll results → {OUT_ROOT}")


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER  (LSTM — same as zirvae_multifile.py)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=64, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def train_lstm(X_tr, y_tr, device='cpu', epochs=30, batch_size=128):
    F_  = X_tr.shape[2]
    clf = LSTMClassifier(F_).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    crt = nn.CrossEntropyLoss()
    Xt  = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt  = torch.tensor(y_tr, dtype=torch.long,    device=device)
    ldr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt),
        batch_size=batch_size, shuffle=True)
    clf.train()
    for _ in tqdm(range(epochs), desc='      LSTM clf', unit='ep',
                  dynamic_ncols=True, leave=False):
        for xb, yb in ldr:
            opt.zero_grad(); crt(clf(xb), yb).backward(); opt.step()
    clf.eval()
    return clf


def eval_clf(clf, X, y, device='cpu'):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = clf(Xt).argmax(1).cpu().numpy()
    return float(f1_score(y, preds, average='macro', zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ks_per_class(real_X, real_y, syn_X, syn_y, feat_cols):
    results = {}
    for cls in [0, 1]:
        r = real_X[real_y == cls].mean(axis=1)   # mean over T  → (N, F)
        s = syn_X[syn_y  == cls].mean(axis=1)
        rows = []
        for i, col in enumerate(feat_cols):
            stat, p = scipy_stats.ks_2samp(r[:, i], s[:, i])
            rows.append({'feature': col,
                         'ks_stat': round(float(stat), 4),
                         'p_value': round(float(p),    4),
                         'similar': bool(p > 0.05)})
        results[cls] = pd.DataFrame(rows).sort_values('ks_stat', ascending=False)
    return results


def rank_only_f1(X_tr, y_tr, X_te, y_te, feat_cols, device='cpu'):
    """
    Train and evaluate the LSTM on rank and rank.1 ONLY.
    Measures how much of the discriminative signal is in the rank features.
    """
    rank_idx = [feat_cols.index(c) for c in RANK_COLS if c in feat_cols]
    X_tr_r   = X_tr[:, :, rank_idx]
    X_te_r   = X_te[:, :, rank_idx]
    clf      = train_lstm(X_tr_r, y_tr, device=device)
    return eval_clf(clf, X_te_r, y_te, device=device)


def tail_ks_and_coverage(real_X, real_y, syn_X, syn_y,
                         feat_cols, rank_col='rank', quantile=0.10):
    """
    Restrict to windows where rank is BELOW the quantile threshold of the
    normal class (low rank = attack signal) and compute KS + coverage.

    For Blackhole, attack = rank drops down, so we look at LOW rank values
    (bottom 10th percentile of normal rank = most extreme attack signal).
    """
    if rank_col not in feat_cols:
        return pd.DataFrame(), pd.DataFrame()
    ridx = feat_cols.index(rank_col)

    normal_ranks = real_X[real_y == 0, :, ridx].mean(axis=1)
    thresh       = float(np.quantile(normal_ranks, quantile))   # low rank threshold

    def _mask(X, y):
        return (y == 1) & (X[:, :, ridx].mean(axis=1) < thresh)

    r_tail = real_X[_mask(real_X, real_y)]
    s_tail = syn_X[ _mask(syn_X,  syn_y)]

    if len(r_tail) < 5 or len(s_tail) < 5:
        return pd.DataFrame(), pd.DataFrame()

    r_flat = r_tail.mean(axis=1)
    s_flat = s_tail.mean(axis=1)

    ks_rows, cov_rows = [], []
    for i, col in enumerate(feat_cols):
        stat, p  = scipy_stats.ks_2samp(r_flat[:, i], s_flat[:, i])
        ks_rows.append({'feature': col,
                        'tail_ks': round(float(stat), 4),
                        'tail_p':  round(float(p),    4)})
        s_min = s_flat[:, i].min()
        s_max = s_flat[:, i].max()
        cov   = float(((r_flat[:, i] >= s_min) & (r_flat[:, i] <= s_max)).mean())
        cov_rows.append({'feature': col, 'tail_coverage': round(cov, 4)})

    return pd.DataFrame(ks_rows), pd.DataFrame(cov_rows)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_rank_distributions(real_X, real_y, syn_X, syn_y,
                             feat_cols, variant, out_dir):
    """
    Focused plot: rank and rank.1 distributions for normal vs attack,
    real vs synthetic side-by-side.
    """
    rank_cols_present = [c for c in RANK_COLS if c in feat_cols]
    fig, axes = plt.subplots(2, len(rank_cols_present),
                             figsize=(6 * len(rank_cols_present), 8),
                             constrained_layout=True)
    if len(rank_cols_present) == 1:
        axes = axes.reshape(2, 1)

    for col_idx, col in enumerate(rank_cols_present):
        fidx = feat_cols.index(col)
        for row, (cls, label) in enumerate([(0, 'Normal'), (1, 'Attack')]):
            ax   = axes[row, col_idx]
            r    = real_X[real_y == cls, :, fidx].mean(axis=1)
            s    = syn_X[ syn_y  == cls, :, fidx].mean(axis=1)
            ks_v, _ = scipy_stats.ks_2samp(r, s)

            if r.std() < 1e-8 and s.std() < 1e-8:
                ax.set_title(f'{col} [{label}]\n(constant)', fontsize=10)
                continue
            try:
                ax.hist(r, bins=40, alpha=0.5, color='steelblue',
                        density=True, label='Real')
                ax.hist(s, bins=40, alpha=0.5, color='tomato',
                        density=True, label='Synthetic')
            except Exception:
                pass
            try:
                xs = np.linspace(min(r.min(), s.min()),
                                 max(r.max(), s.max()), 150)
                ax.plot(xs, scipy_stats.gaussian_kde(r)(xs), 'steelblue', lw=2)
                ax.plot(xs, scipy_stats.gaussian_kde(s)(xs), 'tomato',    lw=2)
            except Exception:
                pass
            ax.set_title(f'{col}  [{label}]\nKS={ks_v:.4f}', fontsize=10)
            ax.set_xlabel('Normalised value')
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.legend(fontsize=8)

    fig.suptitle(f'Rank feature distributions — {variant}', fontsize=12)
    path = os.path.join(out_dir, f'rank_distributions_{variant}.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_all_features(real_X, real_y, syn_X, syn_y,
                      feat_cols, variant, out_dir):
    """Distribution overlay for all features, 2 rows (normal/attack)."""
    F_ = len(feat_cols)
    fig, axes = plt.subplots(2, F_, figsize=(2.8 * F_, 7),
                             constrained_layout=True)
    for row, (cls, label) in enumerate([(0, 'Normal'), (1, 'Attack')]):
        r_flat = real_X[real_y == cls].mean(axis=1)
        s_flat = syn_X[syn_y  == cls].mean(axis=1)
        for col_idx, col in enumerate(feat_cols):
            ax = axes[row, col_idx]
            r  = r_flat[:, col_idx]
            s  = s_flat[:, col_idx]
            # skip degenerate arrays (all-zero dead features)
            if r.std() < 1e-8 and s.std() < 1e-8:
                ax.set_title(f'{col}\n(constant)', fontsize=6)
                continue
            try:
                ax.hist(r, bins=30, alpha=0.4, color='steelblue', density=True, label='Real')
                ax.hist(s, bins=30, alpha=0.4, color='tomato',    density=True, label='Syn')
            except Exception:
                pass
            try:
                xs = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 120)
                ax.plot(xs, scipy_stats.gaussian_kde(r)(xs), 'steelblue', lw=1.2)
                ax.plot(xs, scipy_stats.gaussian_kde(s)(xs), 'tomato',    lw=1.2)
            except Exception:
                pass
            ks_v, _ = scipy_stats.ks_2samp(r, s)
            ax.set_title(f'{col}\nKS={ks_v:.3f}', fontsize=6)
            ax.tick_params(labelsize=5)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=7)
            if row == 0 and col_idx == 0:
                ax.legend(fontsize=5)
    fig.suptitle(f'All feature distributions — {variant}', fontsize=9)
    path = os.path.join(out_dir, f'distributions_{variant}.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_ks_bars(ks_results, variant, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for ax, cls in zip(axes, [0, 1]):
        df  = ks_results[cls].sort_values('ks_stat', ascending=False)
        clr = ['steelblue' if r else 'tomato' for r in df['similar']]
        ax.bar(df['feature'], df['ks_stat'], color=clr)
        ax.axhline(0.05, color='black', linestyle='--', lw=1.2, label='p=0.05 threshold')
        label = 'Normal' if cls == 0 else 'Attack'
        ax.set_title(f'Class {cls} ({label})  '
                     f'pass={df["similar"].sum()}/{len(df)}  '
                     f'mean KS={df["ks_stat"].mean():.3f}')
        ax.set_ylabel('KS Statistic'); ax.set_ylim(0, 1.05)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle(f'KS per feature — {variant}  (blue=pass, red=fail)', fontsize=10)
    path = os.path.join(out_dir, f'ks_{variant}.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_ar1_fit(real_X, real_y, syn_X, syn_y, feat_cols,
                 raw_rank_params, variant, out_dir):
    """
    Validate the AR(1) fit: compare lag-1 scatter plot of real vs synthetic
    attack rank sequences (both in preprocessed [0,1] space).
    The AR(1) was fit in raw space; here we show the empirical lag-1 correlation
    of the synthetic sequences vs real as a validation.
    """
    rank_cols_present = [c for c in RANK_COLS if c in feat_cols
                         if c in raw_rank_params]
    if not rank_cols_present:
        return
    fig, axes = plt.subplots(2, len(rank_cols_present),
                             figsize=(6 * len(rank_cols_present), 10),
                             constrained_layout=True)
    if len(rank_cols_present) == 1:
        axes = axes.reshape(2, 1)

    for col_idx, col in enumerate(rank_cols_present):
        fidx = feat_cols.index(col)
        p_raw = raw_rank_params[col]['attack']  # raw-space params for annotation
        for row, (X, y, label, color) in enumerate([
            (real_X, real_y, 'Real attack',      'steelblue'),
            (syn_X,  syn_y,  'Synthetic attack', 'tomato'),
        ]):
            seqs = X[y == 1, :, fidx]        # (N, T) — preprocessed values
            x_t  = seqs[:, :-1].flatten()
            x_t1 = seqs[:,  1:].flatten()
            # Empirical lag-1 autocorrelation (in preprocessed space)
            if x_t.std() > 1e-6:
                emp_phi = float(np.corrcoef(x_t, x_t1)[0, 1])
            else:
                emp_phi = 0.0
            # subsample for plot clarity
            rng_idx = np.random.default_rng(42)
            idx = rng_idx.choice(len(x_t), min(3000, len(x_t)), replace=False)
            ax  = axes[row, col_idx]
            ax.scatter(x_t[idx], x_t1[idx], alpha=0.15, s=4, color=color)
            # empirical regression line
            xs = np.linspace(x_t.min(), x_t.max(), 100)
            mu_pp = x_t.mean()
            ys = emp_phi * (xs - mu_pp) + mu_pp
            ax.plot(xs, ys, 'k--', lw=1.5,
                    label=f"emp φ={emp_phi:.3f} | raw μ={p_raw['mu']:.1f}")
            ax.set_xlabel(f'{col}[t] (preprocessed)', fontsize=8)
            ax.set_ylabel(f'{col}[t+1] (preprocessed)', fontsize=8)
            ax.set_title(f'{label} — {col}', fontsize=9)
            ax.legend(fontsize=7)

    fig.suptitle(f'AR(1) validation (lag-1 scatter, preprocessed space) — {variant}',
                 fontsize=11)
    path = os.path.join(out_dir, f'ar1_validation_{variant}.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_summary(summary_rows, out_dir):
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return

    variants = df['variant'].tolist()
    x        = np.arange(len(variants))
    w        = 0.25

    # TSTR / TRTS / Baseline
    fig, ax = plt.subplots(figsize=(max(10, len(variants)), 5),
                           constrained_layout=True)
    ax.bar(x - w, df['baseline_f1'], w, label='Baseline (real→real)',
           color='steelblue', alpha=0.85)
    ax.bar(x,     df['tstr_f1'],     w, label='TSTR (syn→real)',
           color='tomato',    alpha=0.85)
    ax.bar(x + w, df['trts_f1'],     w, label='TRTS (real→syn)',
           color='seagreen',  alpha=0.85)
    for i, row in df.iterrows():
        for off, val in [(-w, row['baseline_f1']),
                         (0,  row['tstr_f1']),
                         (w,  row['trts_f1'])]:
            ax.text(i + off, val + 0.01, f'{val:.2f}',
                    ha='center', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Macro F1')
    ax.axhline(0.5, color='red', linestyle='--', lw=1, label='random')
    ax.legend(fontsize=8)
    ax.set_title('TSTR / TRTS / Baseline — All Blackhole Variants (Perturbation)')
    fig.savefig(os.path.join(out_dir, 'tstr_trts_all_variants.png'),
                dpi=120, bbox_inches='tight')
    plt.close(fig)

    # KS mean — cls0 and cls1
    fig, ax = plt.subplots(figsize=(max(10, len(variants)), 4),
                           constrained_layout=True)
    ax.plot(variants, df['ks_cls0_mean'], 'o-', color='steelblue', label='Normal KS mean')
    ax.plot(variants, df['ks_cls1_mean'], 's-', color='tomato',    label='Attack KS mean')
    ax.axhline(0.05, color='green', linestyle='--', lw=1, label='p=0.05 line')
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8); ax.set_ylabel('Mean KS')
    ax.set_title('Mean KS across features — All Variants (Perturbation)')
    fig.savefig(os.path.join(out_dir, 'ks_trend.png'),
                dpi=120, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PER-VARIANT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(variant: str) -> dict:
    tqdm.write(f"\n{'='*65}")
    tqdm.write(f"  VARIANT: {variant}")
    tqdm.write(f"{'='*65}")

    folder_path = os.path.join(DATA_ROOT, variant)
    out_dir     = os.path.join(OUT_ROOT, variant)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # ── Load & preprocess ────────────────────────────────────────────────────
    df_tr, df_te, df_tr_raw, df_te_raw = split_files(folder_path)
    (X_tr, y_tr,
     X_te, y_te,
     feat_cols, active_cols, preproc) = preprocess_and_window(df_tr, df_te)
    T, F_full   = X_tr.shape[1], X_tr.shape[2]
    F_active    = len(active_cols)
    active_idx  = [feat_cols.index(c) for c in active_cols]

    tqdm.write(f"  Train: {X_tr.shape}  Test: {X_te.shape}")
    tqdm.write(f"  Label dist train: "
               f"{dict(zip(*np.unique(y_tr, return_counts=True)))}")
    tqdm.write(f"  Active features ({F_active}): {active_cols}")
    tqdm.write(f"  Dead features excluded: {DEAD_FEATURES}")

    # Active-feature-only arrays for VAE
    X_tr_active = X_tr[:, :, active_idx]   # (N, T, F_active)
    X_te_active = X_te[:, :, active_idx]

    # ── Hyperparams ──────────────────────────────────────────────────────────
    latent_dim = max(8, min((T * F_active) // 8, 32))
    hidden_dim = max(64, F_active * 6)
    batch_size = min(512, max(64, len(X_tr) // 10))
    free_bits  = round(min(0.5, 8.0 / latent_dim), 4)
    tqdm.write(f"  latent={latent_dim}  hidden={hidden_dim}  "
               f"batch={batch_size}  free_bits={free_bits}")

    # ── Baseline LSTM ─────────────────────────────────────────────────────────
    tqdm.write(f"\n  [1/4] Baseline LSTM (all features) …")
    base_clf = train_lstm(X_tr, y_tr, device=DEVICE)
    base_f1  = eval_clf(base_clf, X_te, y_te, device=DEVICE)
    tqdm.write(f"       Baseline F1 = {base_f1:.4f}")

    # Rank-only baseline: how much signal is in rank alone?
    tqdm.write(f"\n  [1b]  Rank-only LSTM …")
    rank_f1 = rank_only_f1(X_tr, y_tr, X_te, y_te, feat_cols, device=DEVICE)
    tqdm.write(f"       Rank-only F1 = {rank_f1:.4f}")

    # ── Train ZICVAE on all windows (both classes) ───────────────────────────
    tqdm.write(f"\n  [2/4] ZICVAE ({VAE_EPOCHS} epochs, "
               f"{F_active} active features, both classes) …")
    vae = train_zicvae(
        X_tr_active, y_tr, T, active_cols,
        hidden_dim=hidden_dim, latent_dim=latent_dim,
        epochs=VAE_EPOCHS, batch_size=batch_size,
        noise_std=0.05, free_bits=free_bits,
        device=DEVICE,
    )

    # ── Fit AR(1) perturbation parameters on RAW rank ────────────────────────
    tqdm.write(f"\n  [3/4] Fitting AR(1) on raw rank (level signal preserved) …")
    raw_rank_params = fit_rank_ar1_raw(df_tr_raw)

    # ── Generate synthetic data ───────────────────────────────────────────────
    tqdm.write(f"\n  [4/4] Generating {N_SYNTH} synthetic windows per class …")

    # Normal: CVAE sample with y=0, embed back into full feature space
    syn_active_normal = vae.sample(N_SYNTH, y_val=0, device=DEVICE)  # (N, T, F_active)
    syn_full_normal   = np.zeros((N_SYNTH, T, F_full), dtype=np.float32)
    for i, aidx in enumerate(active_idx):
        syn_full_normal[:, :, aidx] = syn_active_normal[:, :, i]
    # dead features stay 0 (Bernoulli all-zero — correct for Blackhole)

    # Attack background: CVAE sample with y=1 — non-rank features now drawn
    # from P(features | attack) instead of the normal-class distribution
    syn_active_attack_bg = vae.sample(N_SYNTH, y_val=1, device=DEVICE)
    syn_full_attack_bg   = np.zeros((N_SYNTH, T, F_full), dtype=np.float32)
    for i, aidx in enumerate(active_idx):
        syn_full_attack_bg[:, :, aidx] = syn_active_attack_bg[:, :, i]

    # Overwrite rank and rank.1 with AR(1) sequences (unchanged)
    syn_full_attack = generate_attack_windows(
        syn_full_attack_bg, raw_rank_params, preproc, feat_cols, rng)

    X_syn = np.concatenate([syn_full_normal, syn_full_attack], axis=0)
    y_syn = np.concatenate([
        np.zeros(N_SYNTH, dtype=np.int64),
        np.ones( N_SYNTH, dtype=np.int64),
    ])

    # ── Evaluate ──────────────────────────────────────────────────────────────
    ks_res  = ks_per_class(X_tr, y_tr, X_syn, y_syn, feat_cols)
    tstr_f1 = eval_clf(train_lstm(X_syn, y_syn, device=DEVICE),
                       X_te,  y_te,  device=DEVICE)
    trts_f1 = eval_clf(train_lstm(X_tr,  y_tr,  device=DEVICE),
                       X_syn, y_syn, device=DEVICE)

    tail_ks_df, tail_cov_df = tail_ks_and_coverage(
        X_tr, y_tr, X_syn, y_syn, feat_cols,
        rank_col='rank', quantile=0.10)

    # Conditional precision: real clf scores synthetic attacks
    clf_real = train_lstm(X_tr, y_tr, device=DEVICE)
    Xt_atk   = torch.tensor(syn_full_attack, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        preds_atk = clf_real(Xt_atk).argmax(1).cpu().numpy()
    cond_precision = float((preds_atk == 1).mean())

    # Print
    for cls in [0, 1]:
        kdf   = ks_res[cls]
        label = 'Normal' if cls == 0 else 'Attack'
        tqdm.write(f"  KS cls{cls} ({label}): "
                   f"pass={kdf['similar'].sum()}/{len(kdf)}  "
                   f"mean_KS={kdf['ks_stat'].mean():.4f}")
    tqdm.write(f"  TSTR={tstr_f1:.4f}  TRTS={trts_f1:.4f}  "
               f"Baseline={base_f1:.4f}  RankOnly={rank_f1:.4f}")
    if not tail_ks_df.empty:
        tqdm.write(f"  Mean Tail KS (low rank):       "
                   f"{tail_ks_df['tail_ks'].mean():.4f}")
        tqdm.write(f"  Mean Tail Coverage (low rank): "
                   f"{tail_cov_df['tail_coverage'].mean():.4f}")
    tqdm.write(f"  Cond. Precision: {cond_precision:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_rank_distributions(X_tr, y_tr, X_syn, y_syn,
                            feat_cols, variant, out_dir)
    plot_all_features(X_tr, y_tr, X_syn, y_syn,
                      feat_cols, variant, out_dir)
    plot_ks_bars(ks_res, variant, out_dir)
    plot_ar1_fit(X_tr, y_tr, X_syn, y_syn,
                 feat_cols, raw_rank_params, variant, out_dir)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        'variant':             variant,
        'baseline_f1':         round(base_f1,  4),
        'rank_only_f1':        round(rank_f1,  4),
        'tstr_f1':             round(tstr_f1,  4),
        'trts_f1':             round(trts_f1,  4),
        'cond_precision':      round(cond_precision, 4),
        'ks_cls0_mean':        round(float(ks_res[0]['ks_stat'].mean()), 4),
        'ks_cls1_mean':        round(float(ks_res[1]['ks_stat'].mean()), 4),
        'ks_cls0_pass':        int(ks_res[0]['similar'].sum()),
        'ks_cls1_pass':        int(ks_res[1]['similar'].sum()),
        'mean_tail_ks':        round(float(tail_ks_df['tail_ks'].mean()), 4)
                               if not tail_ks_df.empty else None,
        'mean_tail_coverage':  round(float(tail_cov_df['tail_coverage'].mean()), 4)
                               if not tail_cov_df.empty else None,
        'ar1_params':          {
            col: {k: {kk: round(vv, 6) for kk, vv in v.items()}
                  for k, v in p_dict.items()}
            for col, p_dict in raw_rank_params.items()
        },
        'ks_per_feature_cls0': ks_res[0].to_dict(orient='records'),
        'ks_per_feature_cls1': ks_res[1].to_dict(orient='records'),
        'tail_ks':             tail_ks_df.to_dict(orient='records') if not tail_ks_df.empty else [],
        'tail_coverage':       tail_cov_df.to_dict(orient='records') if not tail_cov_df.empty else [],
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Perturbation-based Blackhole synthesis — {len(VARIANTS)} variants")
    all_metrics = []

    outer = tqdm(VARIANTS, desc='Overall', unit='variant',
                 dynamic_ncols=True, position=0)
    for variant in outer:
        outer.set_description(f'Overall [{variant}]')
        try:
            m = run_variant(variant)
            all_metrics.append(m)
            outer.set_postfix(
                tstr=f"{m['tstr_f1']:.3f}",
                ks1=f"{m['ks_cls1_mean']:.3f}")
        except Exception as e:
            import traceback
            tqdm.write(f"  ERROR in {variant}: {e}")
            traceback.print_exc()

    if not all_metrics:
        print("No variants processed.")
        return

    # Save outputs
    with open(os.path.join(OUT_ROOT, 'summary.json'), 'w') as fh:
        json.dump(all_metrics, fh, indent=2)

    flat = [{k: v for k, v in m.items() if not isinstance(v, (list, dict))}
            for m in all_metrics]
    summary_df = pd.DataFrame(flat)
    summary_df.to_csv(os.path.join(OUT_ROOT, 'summary.csv'), index=False)

    plot_summary(all_metrics, OUT_ROOT)

    # Console summary
    print('\n' + '='*80)
    print('  FINAL SUMMARY — Perturbation-based Blackhole Synthesis')
    print('='*80)
    hdr = (f"  {'Variant':<28} {'Base':>6} {'RnkOnly':>8} "
           f"{'TSTR':>6} {'TRTS':>6} {'KS0':>6} {'KS1':>6} {'CondPrec':>9}")
    print(hdr); print('-' * len(hdr))
    for m in all_metrics:
        print(f"  {m['variant']:<28} {m['baseline_f1']:>6.3f} "
              f"{m['rank_only_f1']:>8.3f} "
              f"{m['tstr_f1']:>6.3f} {m['trts_f1']:>6.3f} "
              f"{m['ks_cls0_mean']:>6.3f} {m['ks_cls1_mean']:>6.3f} "
              f"{m['cond_precision']:>9.3f}")

    print(f"\nAll results → {OUT_ROOT}")


if __name__ == '__main__':
    if '--zirvae' in sys.argv:
        main_zirvae()
    else:
        main()
