# disflooding_perturb.py
#
# Perturbation-based synthetic data generation for Disflooding attacks.
#
# Core insight (from data analysis)
# ──────────────────────────────────
# Disflooding's discriminative signal is the OPPOSITE of Blackhole:
#
#   Normal  rank/rank.1:  KS ≈ 0.03,  RF importance ≈ 0.1%  → no signal
#   Attack  disr/diss/dior/dios/tots (+ .1 variants):
#           KS > 0.93,  combined RF importance ≈ 99%
#
#   In normal traffic these flooding traffic features are essentially 0.
#   In attack traffic they jump to large positive values:
#     disr  normal≈0       attack≈18  (+∞)
#     diss  normal≈0       attack≈6
#     dior  normal≈0.58    attack≈16
#     dios  normal≈0.40    attack≈4
#     tots  normal≈0.43    attack≈10
#     (likewise for .1 variants)
#
#   diar / diar.1: KS ≈ 0.055, RF importance ≈ 0%  → effectively dead
#   rank / rank.1: no class-level difference  → background noise only
#
# Strategy
# ─────────
# Normal class:
#   Train a ZICVAE on all windows (both classes) conditioning on label.
#   Sample N synthetic normal windows with y=0.
#
# Attack class  — NO generative model for signal features, pure perturbation:
#   Step 1  Fit AR(1) parameters to REAL attack signal sequences (raw values):
#             phi        = measured lag-1 autocorrelation
#             mu_atk     = attack marginal mean
#             sigma_inn  = innovation std = std_atk * sqrt(1 - phi^2)
#   Step 2  For each synthetic attack window:
#             a. Draw a synthetic attack background from the ZICVAE (y=1)
#             b. Replace all SIGNAL_COLS with AR(1) samples in raw space,
#                then apply log1p + min-max normalisation (same as preproc)
#             c. Clip to [0, 1]
#
# Evaluation
# ──────────
#   Standard: KS per feature (cls 0 and 1), TSTR, TRTS, Baseline F1
#   Signal-specific: signal-only KS, signal-conditional F1
#   Tail: tail KS, tail coverage  (windows where signal features are HIGH)
#
# Run:
#   conda run -n vinnova python toy/disflooding_perturb.py
#
# Output:
#   toy/disflooding_perturb_results/<variant>/
#   toy/disflooding_perturb_results/summary.csv
#   toy/disflooding_perturb_results/summary.json

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
                          'disflooding_perturb_results')
os.makedirs(OUT_ROOT, exist_ok=True)

VARIANTS      = [d for d in sorted(os.listdir(DATA_ROOT)) if d.startswith('disflooding_')]
WINDOW_SIZE   = 10
N_SYNTH       = 1000          # synthetic windows per class
VAE_EPOCHS    = 300
DEVICE        = 'cpu'
RNG_SEED      = 42
SPARSE_THRESH = 0.30

# Primary discriminative features for Disflooding (flooding traffic counters)
SIGNAL_COLS = ['disr', 'diss', 'dior', 'dios', 'tots',
               'disr.1', 'diss.1', 'dior.1', 'dios.1', 'tots.1']

# diar / diar.1: KS ≈ 0.055, RF importance ≈ 0% — excluded from VAE input
DEAD_FEATURES = ['diar', 'diar.1']


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (identical split logic to blackhole_perturb.py)
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

    n_train   = max(1, int(len(shuffled) * 0.7))
    df_tr     = _load(shuffled[:n_train])
    df_te     = _load(shuffled[n_train:]) if len(shuffled) > n_train else _load(shuffled[-1:])
    # Keep raw copies (before any preprocessing) for AR(1) fitting on signal levels
    df_tr_raw = df_tr.copy()
    df_te_raw = df_te.copy()
    return df_tr, df_te, df_tr_raw, df_te_raw


def preprocess_and_window(df_tr: pd.DataFrame, df_te: pd.DataFrame):
    """
    Preprocessing fitted on train.

    Disflooding's attack signal is in the LEVEL of the flooding traffic features
    (disr, diss, dior, dios, tots and .1 variants).  These features are near-zero
    for normal traffic and large for attack traffic.  We preserve this level by
    NOT differencing and using log1p + min-max normalization so the zero-inflation
    (normal class) and high-value tail (attack class) are both captured.

    Steps:
      1. log1p on sparse cols (zero% > SPARSE_THRESH) — includes signal cols
      2. Min-max normalisation → [0, 1] on all features

    Returns:
      X_tr, y_tr, X_te, y_te  — windowed arrays  (N, T, F)
      feat_cols                — list of all feature names
      active_cols              — feat_cols minus DEAD_FEATURES
      preproc                  — dict of fitted scalers
    """
    feat_cols     = [c for c in df_tr.columns if c != 'label']
    signal_present = [c for c in SIGNAL_COLS if c in feat_cols]

    # 1. log1p on sparse cols (includes signal cols — zero in normal, positive in attack)
    sparse_cols = [c for c in feat_cols
                   if (df_tr[c] == 0).mean() > SPARSE_THRESH]
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
                   sparse_cols=sparse_cols, signal_present=signal_present)

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


def fit_signal_ar1_raw(df_tr_raw: pd.DataFrame) -> dict:
    """
    Fit AR(1) on RAW (unprocessed) signal feature values, for the attack class.

    Disflooding signal:
      Normal  disr/diss/dior/dios/tots: ≈ 0 (all-zero or near-zero)
      Attack  disr: mean ≈ 18,  diss: ≈ 6,  dior: ≈ 16,  dios: ≈ 4,  tots: ≈ 10
              (similarly for .1 variants)

    The zero→high jump is the entire discriminative signal.
    Fitting on preprocessed space is fine since log1p preserves the AR structure,
    but we fit on raw values and apply the same preprocessing chain when converting.

    Returns:
      { col: { 'normal': {mu, std, phi, sigma_inn},
               'attack': {mu, std, phi, sigma_inn} } }
    """
    params = {}
    for col in SIGNAL_COLS:
        if col not in df_tr_raw.columns:
            continue
        col_params = {}
        for cls, cls_name in [(0, 'normal'), (1, 'attack')]:
            vals = df_tr_raw.loc[df_tr_raw['label'] == cls, col].values.astype(np.float64)
            n_win = len(vals) // WINDOW_SIZE
            if n_win == 0:
                col_params[cls_name] = {
                    'mu': float(vals.mean()), 'std': max(float(vals.std()), 1e-3),
                    'phi': 0.5, 'sigma_inn': max(float(vals.std()), 1e-3) * 0.87}
                continue
            seqs = vals[:n_win * WINDOW_SIZE].reshape(n_win, WINDOW_SIZE)
            col_params[cls_name] = fit_ar1(seqs)
        params[col] = col_params
        pn = col_params['normal']
        pa = col_params['attack']
        tqdm.write(f"    RAW AR(1) {col}: "
                   f"normal  mu={pn['mu']:.2f} std={pn['std']:.2f} phi={pn['phi']:.4f} | "
                   f"attack  mu={pa['mu']:.2f} std={pa['std']:.2f} phi={pa['phi']:.4f}")
    return params


def raw_signal_seq_to_preprocessed(raw_seqs: np.ndarray, col: str,
                                    preproc: dict) -> np.ndarray:
    """
    Apply the signal feature preprocessing chain (log1p + min-max) to raw sequences.

    raw_seqs : (N, T)  — raw signal values
    Returns  : (N, T)  — normalised values in [0, 1]
    """
    g_min_col = float(preproc['g_min'][col])
    denom_col = float(preproc['denom'][col])

    # Apply log1p (same as preproc step 1 — signal cols are sparse)
    logged = np.log1p(np.clip(raw_seqs.astype(np.float64), 0.0, None))

    # Min-max normalise (denom already computed on log1p-transformed train data)
    out = np.clip((logged - g_min_col) / denom_col, 0.0, 1.0).astype(np.float32)
    return out


def sample_ar1(params: dict, n_windows: int, window_size: int,
               rng: np.random.Generator) -> np.ndarray:
    """
    Sample n_windows independent AR(1) sequences of length window_size.

    Each sequence starts by drawing x_0 ~ N(mu, std) (from the marginal),
    then evolves as:  x_t = phi*(x_{t-1} - mu) + mu + N(0, sigma_inn^2)

    Returns array of shape (n_windows, window_size), clipped to >= 0
    (raw signal values are non-negative).
    """
    mu        = params['mu']
    phi       = params['phi']
    sigma_inn = params['sigma_inn']
    std       = params['std']

    out = np.empty((n_windows, window_size), dtype=np.float64)
    out[:, 0] = rng.normal(mu, std, size=n_windows)

    for t in range(1, window_size):
        noise     = rng.normal(0, sigma_inn, size=n_windows)
        out[:, t] = phi * (out[:, t-1] - mu) + mu + noise

    # Raw signal values are non-negative
    return np.clip(out, 0.0, None).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL-CLASS VAE  (trained on active features only)
# ─────────────────────────────────────────────────────────────────────────────

def _kl_weight(epoch, n_epochs, n_cycles=4):
    cycle_len = max(n_epochs / n_cycles, 1)
    return min(1.0, ((epoch % cycle_len) / cycle_len) * 2.0)


class ZICVAE(nn.Module):
    """
    Zero-Inflated Conditional GRU-VAE.

    Identical to the one in blackhole_perturb.py — class label y ∈ {0, 1} is
    fed to both encoder and decoder via a small learned embedding (label_dim=4),
    allowing the model to learn separate conditional distributions for each class.

    For Disflooding, the signal features (disr, diss, dior, dios, tots, .1 variants)
    are zero-inflated — they are always 0 for normal traffic (y=0) and positive for
    attack traffic (y=1).  The CVAE conditioning ensures that normal-class samples
    have near-zero signal features and attack-class backgrounds have the correct
    non-signal feature distributions (rank, etc.).

    The AR(1) perturbation step then overwrites the signal features with sequences
    that match the true attack distribution.
    """
    LABEL_DIM = 4

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
        self.label_emb = nn.Embedding(2, ld)

        self.enc_gru   = nn.GRU(n_features + ld, hidden_dim, n_layers,
                                batch_first=True)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_lv     = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.fc_lv.bias, -1.0)

        zld = latent_dim + ld
        self.fc_h0    = nn.Linear(zld, n_layers * hidden_dim)
        self.fc_input = nn.Linear(zld, n_features)
        self.gru      = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_features)
        self.fc_gate  = nn.Linear(hidden_dim, len(zi_idx)) if zi_idx else None

    def _encode(self, x, y):
        e   = self.label_emb(y).unsqueeze(1).expand(-1, x.size(1), -1)
        _, h = self.enc_gru(torch.cat([x, e], dim=-1))
        h    = h[-1]
        return self.fc_mu(h), self.fc_lv(h)

    def _decode(self, z, y):
        B   = z.size(0)
        e   = self.label_emb(y)
        zy  = torch.cat([z, e], dim=-1)
        h0  = self.fc_h0(zy).view(self.n_layers, B, self.hidden_dim)
        inp = self.fc_input(zy).unsqueeze(1).expand(-1, self.window_size, -1)
        gru_out, _ = self.gru(inp, h0)
        out        = torch.sigmoid(self.fc_out(gru_out))
        gate       = self.fc_gate(gru_out) if self.fc_gate else None
        return out, gate

    def forward(self, x, y):
        mu, lv    = self._encode(x, y)
        z         = REncoder.reparameterize(mu, lv)
        out, gate = self._decode(z, y)
        return out, gate, mu, lv

    @torch.no_grad()
    def sample(self, n, y_val: int, device='cpu'):
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


def _vae_loss(x, out, gate, mu, lv, zi_idx, lognorm_idx, signal_idx,
              kl_weight, free_bits):
    """
    Mixed loss:
      signal features → MSE  (continuous after log1p normalisation)
      lognorm features → MSE on log1p-normalised values
      zi features     → gate BCE + MSE on non-zero values
    KL: free-bits with cyclical annealing weight.
    """
    recon   = torch.tensor(0.0, device=x.device)
    n_terms = 0

    for f in signal_idx + lognorm_idx:
        recon   += F_torch.mse_loss(out[:, :, f], x[:, :, f])
        n_terms += 1

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
    """Train ZICVAE on all windows (both classes) with class labels."""
    F_  = len(feat_cols_active)
    if latent_dim is None:
        latent_dim = max(8, min((window_size * F_) // 8, 32))

    X_normal    = X_all[y_all == 0]
    # Signal cols: perturbed by AR(1) — treated like rank in blackhole
    signal_idx  = [i for i, c in enumerate(feat_cols_active) if c in SIGNAL_COLS]
    zi_idx      = [i for i, c in enumerate(feat_cols_active)
                   if c not in SIGNAL_COLS and
                   (X_normal[:, :, i] == 0).mean() > SPARSE_THRESH]
    lognorm_idx = [i for i in range(F_)
                   if i not in signal_idx and i not in zi_idx]

    tqdm.write(f"    ZICVAE feature buckets:")
    tqdm.write(f"      signal  : {[feat_cols_active[i] for i in signal_idx]}")
    tqdm.write(f"      lognorm : {[feat_cols_active[i] for i in lognorm_idx]}")
    tqdm.write(f"      zi      : {[feat_cols_active[i] for i in zi_idx]}")
    tqdm.write(f"      latent={latent_dim}  hidden={hidden_dim}  batch={batch_size}")
    tqdm.write(f"      training on {len(X_all)} windows "
               f"({(y_all==0).sum()} normal, {(y_all==1).sum()} attack)")

    X_t = torch.tensor(X_all, device=device)
    y_t = torch.tensor(y_all, dtype=torch.long, device=device)
    mdl = ZICVAE(F_, window_size, hidden_dim, latent_dim, zi_idx).to(device)
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
            for s_global in zi_idx:
                zero_mask = (batch[:, :, s_global] == 0)
                noisy[:, :, s_global][zero_mask] = 0.0
            out, gate, mu_z, lv = mdl(noisy, labels)
            loss, _ = _vae_loss(batch, out, gate, mu_z, lv,
                                 zi_idx, lognorm_idx, signal_idx,
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
                             raw_signal_params: dict,
                             preproc: dict,
                             feat_cols: list,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Create synthetic attack windows by:
      1. Starting from a CVAE attack-class background (y=1 samples).
         Non-signal features (rank, rank.1) are drawn from P(features | attack)
         learned by the ZICVAE.
      2. Replacing all SIGNAL_COLS with AR(1) sequences sampled in RAW value
         space, then converted to the preprocessed normalised space (log1p + min-max).
         This guarantees the flooding traffic level that is the primary
         Disflooding discriminative signal.
      3. Clipping to [0, 1]

    attack_background : (N, T, F) — CVAE samples with y=1 (preprocessed)
    raw_signal_params : output of fit_signal_ar1_raw()
    preproc           : output of preprocess_and_window()
    Returns           : (N, T, F) — synthetic attack windows (preprocessed)
    """
    N, T, F_ = attack_background.shape
    attack    = attack_background.copy()

    for col, p_dict in raw_signal_params.items():
        if col not in feat_cols:
            continue
        idx      = feat_cols.index(col)
        p_attack = p_dict['attack']

        # Sample AR(1) sequences in RAW signal space
        raw_seqs = sample_ar1(p_attack, N, T, rng)   # (N, T) — raw values >= 0

        # Convert to preprocessed normalised space (log1p + min-max)
        preprocessed = raw_signal_seq_to_preprocessed(raw_seqs, col, preproc)
        attack[:, :, idx] = preprocessed

    return np.clip(attack, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER  (LSTM)
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
        r = real_X[real_y == cls].mean(axis=1)
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


def signal_only_f1(X_tr, y_tr, X_te, y_te, feat_cols, device='cpu'):
    """
    Train and evaluate the LSTM on SIGNAL_COLS only.
    Measures how much discriminative signal is in the flooding traffic features.
    """
    sig_idx = [feat_cols.index(c) for c in SIGNAL_COLS if c in feat_cols]
    if not sig_idx:
        return 0.0
    X_tr_s = X_tr[:, :, sig_idx]
    X_te_s = X_te[:, :, sig_idx]
    clf    = train_lstm(X_tr_s, y_tr, device=device)
    return eval_clf(clf, X_te_s, y_te, device=device)


def tail_ks_and_coverage(real_X, real_y, syn_X, syn_y,
                         feat_cols, signal_col='tots', quantile=0.90):
    """
    Restrict to windows where the primary signal feature (tots) is ABOVE the
    quantile threshold of the attack class (high signal = confirmed attack).

    For Disflooding, attack = flooding traffic is high, so we look at windows
    with the top 10% of tots values in the attack class.
    """
    if signal_col not in feat_cols:
        # Fall back to first available signal col
        available = [c for c in SIGNAL_COLS if c in feat_cols]
        if not available:
            return pd.DataFrame(), pd.DataFrame()
        signal_col = available[0]

    sidx = feat_cols.index(signal_col)

    attack_signal = real_X[real_y == 1, :, sidx].mean(axis=1)
    thresh        = float(np.quantile(attack_signal, quantile))

    def _mask(X, y):
        return (y == 1) & (X[:, :, sidx].mean(axis=1) > thresh)

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

def plot_signal_distributions(real_X, real_y, syn_X, syn_y,
                               feat_cols, variant, out_dir):
    """
    Focused plot: signal feature distributions for normal vs attack,
    real vs synthetic side-by-side.  Shows the top-4 signal cols by KS.
    """
    sig_present = [c for c in SIGNAL_COLS if c in feat_cols]
    if not sig_present:
        return None
    show_cols = sig_present[:4]   # top 4 to keep the figure manageable

    fig, axes = plt.subplots(2, len(show_cols),
                             figsize=(5 * len(show_cols), 8),
                             constrained_layout=True)
    if len(show_cols) == 1:
        axes = axes.reshape(2, 1)

    for col_idx, col in enumerate(show_cols):
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

    fig.suptitle(f'Signal feature distributions — {variant}', fontsize=12)
    path = os.path.join(out_dir, f'signal_distributions_{variant}.png')
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
                 raw_signal_params, variant, out_dir):
    """
    Validate the AR(1) fit for the top-2 signal cols: lag-1 scatter of real
    vs synthetic attack sequences (preprocessed [0,1] space).
    """
    sig_cols_present = [c for c in SIGNAL_COLS if c in feat_cols
                        and c in raw_signal_params][:2]
    if not sig_cols_present:
        return

    fig, axes = plt.subplots(2, len(sig_cols_present),
                             figsize=(6 * len(sig_cols_present), 10),
                             constrained_layout=True)
    if len(sig_cols_present) == 1:
        axes = axes.reshape(2, 1)

    for col_idx, col in enumerate(sig_cols_present):
        fidx  = feat_cols.index(col)
        p_raw = raw_signal_params[col]['attack']
        for row, (X, y, label, color) in enumerate([
            (real_X, real_y, 'Real attack',      'steelblue'),
            (syn_X,  syn_y,  'Synthetic attack', 'tomato'),
        ]):
            seqs = X[y == 1, :, fidx]
            x_t  = seqs[:, :-1].flatten()
            x_t1 = seqs[:,  1:].flatten()
            emp_phi = float(np.corrcoef(x_t, x_t1)[0, 1]) if x_t.std() > 1e-6 else 0.0

            rng_idx = np.random.default_rng(42)
            idx = rng_idx.choice(len(x_t), min(3000, len(x_t)), replace=False)
            ax  = axes[row, col_idx]
            ax.scatter(x_t[idx], x_t1[idx], alpha=0.15, s=4, color=color)
            xs = np.linspace(x_t.min(), x_t.max(), 100)
            mu_pp = x_t.mean()
            ax.plot(xs, emp_phi * (xs - mu_pp) + mu_pp, 'k--', lw=1.5,
                    label=f"emp φ={emp_phi:.3f} | raw μ={p_raw['mu']:.2f}")
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
            ax.text(i + off, val + 0.01, f'{val:.2f}', ha='center', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Macro F1')
    ax.axhline(0.5, color='red', linestyle='--', lw=1, label='random')
    ax.legend(fontsize=8)
    ax.set_title('TSTR / TRTS / Baseline — All Disflooding Variants (Perturbation)')
    fig.savefig(os.path.join(out_dir, 'tstr_trts_all_variants.png'),
                dpi=120, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(variants)), 4),
                           constrained_layout=True)
    ax.plot(variants, df['ks_cls0_mean'], 'o-', color='steelblue', label='Normal KS mean')
    ax.plot(variants, df['ks_cls1_mean'], 's-', color='tomato',    label='Attack KS mean')
    ax.axhline(0.05, color='green', linestyle='--', lw=1, label='p=0.05 line')
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8); ax.set_ylabel('Mean KS')
    ax.set_title('Mean KS across features — All Variants (Perturbation)')
    fig.savefig(os.path.join(out_dir, 'ks_trend.png'), dpi=120, bbox_inches='tight')
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

    X_tr_active = X_tr[:, :, active_idx]
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

    # Signal-only baseline: how much signal is in flooding features alone?
    tqdm.write(f"\n  [1b]  Signal-only LSTM …")
    sig_f1 = signal_only_f1(X_tr, y_tr, X_te, y_te, feat_cols, device=DEVICE)
    tqdm.write(f"       Signal-only F1 = {sig_f1:.4f}")

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

    # ── Fit AR(1) perturbation parameters on RAW signal features ─────────────
    tqdm.write(f"\n  [3/4] Fitting AR(1) on raw signal features …")
    raw_signal_params = fit_signal_ar1_raw(df_tr_raw)

    # ── Generate synthetic data ───────────────────────────────────────────────
    tqdm.write(f"\n  [4/4] Generating {N_SYNTH} synthetic windows per class …")

    # Normal: CVAE sample with y=0, embed into full feature space
    syn_active_normal = vae.sample(N_SYNTH, y_val=0, device=DEVICE)
    syn_full_normal   = np.zeros((N_SYNTH, T, F_full), dtype=np.float32)
    for i, aidx in enumerate(active_idx):
        syn_full_normal[:, :, aidx] = syn_active_normal[:, :, i]
    # dead features (diar, diar.1) stay 0

    # Attack background: CVAE sample with y=1
    syn_active_attack_bg = vae.sample(N_SYNTH, y_val=1, device=DEVICE)
    syn_full_attack_bg   = np.zeros((N_SYNTH, T, F_full), dtype=np.float32)
    for i, aidx in enumerate(active_idx):
        syn_full_attack_bg[:, :, aidx] = syn_active_attack_bg[:, :, i]

    # Overwrite signal cols with AR(1) sequences
    syn_full_attack = generate_attack_windows(
        syn_full_attack_bg, raw_signal_params, preproc, feat_cols, rng)

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
        signal_col='tots', quantile=0.90)

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
               f"Baseline={base_f1:.4f}  SignalOnly={sig_f1:.4f}")
    if not tail_ks_df.empty:
        tqdm.write(f"  Mean Tail KS (high signal):       "
                   f"{tail_ks_df['tail_ks'].mean():.4f}")
        tqdm.write(f"  Mean Tail Coverage (high signal): "
                   f"{tail_cov_df['tail_coverage'].mean():.4f}")
    tqdm.write(f"  Cond. Precision: {cond_precision:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_signal_distributions(X_tr, y_tr, X_syn, y_syn,
                              feat_cols, variant, out_dir)
    plot_all_features(X_tr, y_tr, X_syn, y_syn,
                      feat_cols, variant, out_dir)
    plot_ks_bars(ks_res, variant, out_dir)
    plot_ar1_fit(X_tr, y_tr, X_syn, y_syn,
                 feat_cols, raw_signal_params, variant, out_dir)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        'variant':              variant,
        'baseline_f1':          round(base_f1,  4),
        'signal_only_f1':       round(sig_f1,   4),
        'tstr_f1':              round(tstr_f1,  4),
        'trts_f1':              round(trts_f1,  4),
        'cond_precision':       round(cond_precision, 4),
        'ks_cls0_mean':         round(float(ks_res[0]['ks_stat'].mean()), 4),
        'ks_cls1_mean':         round(float(ks_res[1]['ks_stat'].mean()), 4),
        'ks_cls0_pass':         int(ks_res[0]['similar'].sum()),
        'ks_cls1_pass':         int(ks_res[1]['similar'].sum()),
        'mean_tail_ks':         round(float(tail_ks_df['tail_ks'].mean()), 4)
                                if not tail_ks_df.empty else None,
        'mean_tail_coverage':   round(float(tail_cov_df['tail_coverage'].mean()), 4)
                                if not tail_cov_df.empty else None,
        'ar1_params':           {
            col: {k: {kk: round(vv, 6) for kk, vv in v.items()}
                  for k, v in p_dict.items()}
            for col, p_dict in raw_signal_params.items()
        },
        'ks_per_feature_cls0':  ks_res[0].to_dict(orient='records'),
        'ks_per_feature_cls1':  ks_res[1].to_dict(orient='records'),
        'tail_ks':              tail_ks_df.to_dict(orient='records') if not tail_ks_df.empty else [],
        'tail_coverage':        tail_cov_df.to_dict(orient='records') if not tail_cov_df.empty else [],
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Perturbation-based Disflooding synthesis — {len(VARIANTS)} variants")
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

    with open(os.path.join(OUT_ROOT, 'summary.json'), 'w') as fh:
        json.dump(all_metrics, fh, indent=2)

    flat = [{k: v for k, v in m.items() if not isinstance(v, (list, dict))}
            for m in all_metrics]
    summary_df = pd.DataFrame(flat)
    summary_df.to_csv(os.path.join(OUT_ROOT, 'summary.csv'), index=False)

    plot_summary(all_metrics, OUT_ROOT)

    print('\n' + '='*80)
    print('  FINAL SUMMARY — Perturbation-based Disflooding Synthesis')
    print('='*80)
    hdr = (f"  {'Variant':<30} {'Base':>6} {'SigOnly':>8} "
           f"{'TSTR':>6} {'TRTS':>6} {'KS0':>6} {'KS1':>6} {'CondPrec':>9}")
    print(hdr); print('-' * len(hdr))
    for m in all_metrics:
        print(f"  {m['variant']:<30} {m['baseline_f1']:>6.3f} "
              f"{m['signal_only_f1']:>8.3f} "
              f"{m['tstr_f1']:>6.3f} {m['trts_f1']:>6.3f} "
              f"{m['ks_cls0_mean']:>6.3f} {m['ks_cls1_mean']:>6.3f} "
              f"{m['cond_precision']:>9.3f}")

    print(f"\nAll results → {OUT_ROOT}")


if __name__ == '__main__':
    main()
