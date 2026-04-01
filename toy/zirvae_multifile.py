# zirvae_multifile.py
#
# Train a ZI-RVAE generator on 14 files from blackhole_var5_base,
# evaluate on the remaining 6 files.
#
# Training  : files 1–14  (all rows concatenated, per-class generator)
# Testing   : files 15–20 (all rows concatenated, held-out real data)
#
# Metrics:
#   - KS test per feature — class 0 and class 1 separately
#   - TSTR / TRTS / Baseline F1
#   - Distribution overlay plots
#   - Correlation heatmaps
#
# Run:  python toy/zirvae_multifile.py

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F_torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)

from rvae import (REncoder,
                  train_lstm_attn_vae, LSTMAttnVAE,
                  train_transformer_vae, TransformerVAE)

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
DATA_DIR    = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'attack_data', 'blackhole_var20_base'
)
_all_files  = [f'{i}_features_timeseries_60_sec.csv' for i in range(1, 21)]
_rng        = np.random.default_rng(seed=42)   # fixed seed → reproducible split
_shuffled   = _rng.permutation(_all_files).tolist()
TRAIN_FILES = _shuffled[:14]
TEST_FILES  = _shuffled[14:]

WINDOW_SIZE       = 10
N_SYNTH_PER_CLASS = 1000
EPOCHS            = 1000
DEVICE            = 'cpu'
SPARSE_THRESH     = 0.30
RANK_COLS         = ['rank', 'rank.1']

# Thresholds for auto-detecting feature distribution types from training data
BERNOULLI_ZERO_THRESH  = 0.90   # ≥90% zeros + ≤3 unique values → Bernoulli head
ZI_ZERO_THRESH         = 0.30   # ≥30% zeros (and not Bernoulli) → ZI gate
LOGNORMAL_SKEW_THRESH  = 1.0    # skew > 1.0 on non-rank features → log-normal head

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'zirvae_multifile_results')
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════

def load_files(file_list, data_dir):
    """Load and concatenate a list of CSV files. Returns a single DataFrame."""
    dfs = []
    for fname in file_list:
        path = os.path.join(data_dir, fname)
        df   = pd.read_csv(path, encoding='utf-8', encoding_errors='ignore')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def detect_feature_types(df_raw, feat_cols, rank_cols):
    """
    Inspect raw (un-transformed) training data and assign each feature a type:
      'bernoulli'   — near-binary (≥BERNOULLI_ZERO_THRESH zeros, ≤3 unique values)
      'zi_lognorm'  — zero-inflated continuous (≥ZI_ZERO_THRESH zeros, not Bernoulli)
      'lognormal'   — right-skewed continuous (skew > LOGNORMAL_SKEW_THRESH, non-negative)
      'continuous'  — everything else (rank-like, Gaussian after differencing)

    Returns dict {col_name: type_string}
    """
    from scipy.stats import skew as scipy_skew
    types = {}
    for col in feat_cols:
        if col in rank_cols:
            types[col] = 'continuous'
            continue
        x = df_raw[col].dropna().values
        n_unique  = len(np.unique(x))
        zero_pct  = float((x == 0).mean())
        col_skew  = float(scipy_skew(x))

        if zero_pct >= BERNOULLI_ZERO_THRESH and n_unique <= 3:
            types[col] = 'bernoulli'
        elif zero_pct >= ZI_ZERO_THRESH:
            types[col] = 'zi_lognorm'
        elif x.min() >= 0 and col_skew > LOGNORMAL_SKEW_THRESH:
            types[col] = 'lognormal'
        else:
            types[col] = 'continuous'
    return types


def preprocess_and_window(df_tr, df_te, window_size):
    """
    Apply all feature engineering and create windows.

    Feature engineering (fitted on train only):
      1. Differencing + sign-log1p on rank columns
      2. log1p on sparse columns (>SPARSE_THRESH zeros)
      3. Min-max normalization → [0, 1]
      4. Symmetric rescaling for rank columns (0 → 0.5)

    Returns X_tr, y_tr, X_te, y_te, feat_cols, sparse_idx
    """
    feat_cols = [c for c in df_tr.columns if c != 'label']

    # ── 1. Differencing + sign-log1p for rank columns ─
    rank_cols_present = [c for c in RANK_COLS if c in feat_cols]
    for df in [df_tr, df_te]:
        if rank_cols_present:
            diff = df[rank_cols_present].diff().fillna(0)
            df[rank_cols_present] = np.sign(diff) * np.log1p(np.abs(diff))

    # ── 2. Identify sparse columns on train ──────────
    sparse_cols = [
        c for c in feat_cols
        if pd.api.types.is_float_dtype(df_tr[c]) or
           pd.api.types.is_integer_dtype(df_tr[c])
        if (df_tr[c] == 0).mean() > SPARSE_THRESH
    ]
    sparse_idx = [feat_cols.index(c) for c in sparse_cols]
    print(f"   Sparse features ({len(sparse_cols)}): {sparse_cols}")
    print(f"   Rank features differenced: {rank_cols_present}")

    # Apply log1p to sparse cols (non-negative network metrics)
    for df in [df_tr, df_te]:
        if sparse_cols:
            df[sparse_cols] = np.log1p(df[sparse_cols])

    # ── 3. Min-max normalization fitted on train ──────
    g_min  = df_tr[feat_cols].min()
    g_max  = df_tr[feat_cols].max()
    denom  = (g_max - g_min).replace(0, 1)

    for df in [df_tr, df_te]:
        df[feat_cols] = ((df[feat_cols] - g_min) / denom).clip(0, 1).fillna(0)

    # ── 4. Symmetric rescaling for rank cols ──────────
    # After sign-log1p, rank spans [-A, +A] with spike at 0.
    # Min-max maps 0 to interior (~0.56), hiding the spike from the VAE.
    # Fix: scale so 0 → 0.5, negatives → [0, 0.5), positives → (0.5, 1].
    for col in rank_cols_present:
        abs_max = df_tr[col].abs().max()
        if abs_max > 0:
            for df in [df_tr, df_te]:
                df[col] = (df[col] / (2 * abs_max) + 0.5).clip(0, 1)

    # ── 5. Sliding windows ────────────────────────────
    def _windows(df):
        X, y   = [], []
        vals   = df[feat_cols].values.astype(np.float32)
        labels = df['label'].values.astype(int)
        for i in range(len(vals) - window_size):
            window_labels = labels[i:i + window_size]
            # if any row in the window is an attack, label the window as attack
            lbl = 1 if 1 in window_labels else 0
            X.append(vals[i:i + window_size])
            y.append(lbl)
        return np.array(X, np.float32), np.array(y, np.int64)

    X_tr, y_tr = _windows(df_tr)
    X_te, y_te = _windows(df_te)

    return X_tr, y_tr, X_te, y_te, feat_cols, sparse_idx


# ═══════════════════════════════════════════════════
# ZI-RVAE  (mixed-output decoder)
# ═══════════════════════════════════════════════════
#
# feat_types maps each feature index to one of:
#   'bernoulli'   → sigmoid output + BCE loss (no ZI gate needed)
#   'zi_lognorm'  → ZI gate (BCE) + sigmoid μ for non-zero part, MSE in [0,1] space
#   'lognormal'   → sigmoid output + MSE (data already log1p+norm → [0,1])
#   'continuous'  → sigmoid output + MSE (rank cols, after differencing)
#
# The decoder produces ONE shared GRU output per timestep, then routes through
# per-type linear heads.  Bernoulli features get a raw logit head (no sigmoid in
# forward; sigmoid applied only at sample time / in loss).

class ZIRVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_features,
                 window_size, feat_type_idx, n_layers=1):
        """
        feat_type_idx: dict with keys 'bernoulli', 'zi_lognorm', 'lognormal',
                       'continuous', each mapping to a list of feature indices.
        """
        super().__init__()
        self.window_size   = window_size
        self.n_layers      = n_layers
        self.hidden_dim    = hidden_dim
        self.feat_type_idx = feat_type_idx
        self.n_features    = n_features

        self.fc_h0    = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_input = nn.Linear(latent_dim, n_features)
        self.gru      = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True)

        # Shared linear → n_features raw outputs
        self.fc_out = nn.Linear(hidden_dim, n_features)

        # ZI gate head: one logit per zi_lognorm feature
        zi_idx = feat_type_idx.get('zi_lognorm', [])
        self.gate_fc = nn.Linear(hidden_dim, len(zi_idx)) if zi_idx else None

    def forward(self, z):
        B          = z.size(0)
        h0         = self.fc_h0(z).view(self.n_layers, B, self.hidden_dim)
        inp        = self.fc_input(z).unsqueeze(1).expand(-1, self.window_size, -1)
        gru_out, _ = self.gru(inp, h0)
        raw_out    = self.fc_out(gru_out)          # (B, T, F) — raw logits

        # Apply sigmoid for all non-Bernoulli features to keep outputs in [0,1]
        out = torch.sigmoid(raw_out)

        # For Bernoulli features, keep raw logits (loss uses BCE-with-logits)
        bern_idx = self.feat_type_idx.get('bernoulli', [])
        if bern_idx:
            for f in bern_idx:
                out[:, :, f] = raw_out[:, :, f]    # raw logit, NOT sigmoid

        gate_logit = self.gate_fc(gru_out) if self.gate_fc is not None else None
        return out, gate_logit


class ZIRVAE(nn.Module):
    def __init__(self, n_features, window_size, hidden_dim,
                 latent_dim, feat_type_idx, n_layers=1):
        super().__init__()
        self.latent_dim    = latent_dim
        self.feat_type_idx = feat_type_idx
        self.encoder       = REncoder(n_features, hidden_dim, latent_dim, n_layers)
        self.decoder       = ZIRVAEDecoder(latent_dim, hidden_dim, n_features,
                                           window_size, feat_type_idx, n_layers)

    def forward(self, x):
        mu, log_var      = self.encoder(x)
        z                = self.encoder.reparameterize(mu, log_var)
        out, gate_logit  = self.decoder(z)
        return out, gate_logit, mu, log_var

    @torch.no_grad()
    def sample(self, n, device='cpu'):
        z              = torch.randn(n, self.latent_dim, device=device)
        out, gate_logit = self.decoder(z)

        result = out.clone()

        # Bernoulli: convert logit → Bernoulli sample
        for f in self.feat_type_idx.get('bernoulli', []):
            result[:, :, f] = torch.bernoulli(torch.sigmoid(out[:, :, f]))

        # ZI-lognorm: apply gate (already sigmoid-ed output for non-zero part)
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


def _cyclical_kl_weight(epoch, n_epochs, n_cycles=4, min_w=0.0, max_w=1.0):
    cycle_len = max(n_epochs / n_cycles, 1)
    pos       = (epoch % cycle_len) / cycle_len
    ramp      = min(1.0, pos * 2.0)
    return min_w + (max_w - min_w) * ramp


def zirvae_loss(x, decoder_out, gate_logit, mu, log_var,
                feat_type_idx, kl_weight, loss_factor, free_bits):
    """
    Mixed reconstruction loss:
      - Bernoulli features : BCE with logits  (decoder_out holds raw logits)
      - ZI-lognorm features: BCE gate loss  +  MSE on non-zero values in [0,1]
      - Log-normal features : MSE  (data is log1p+norm → [0,1], so MSE ≈ log-space)
      - Continuous features : MSE
    """
    recon = torch.tensor(0.0, device=x.device)
    n_terms = 0

    # Bernoulli head: raw logit in decoder_out → BCE with logits
    for f in feat_type_idx.get('bernoulli', []):
        target = (x[:, :, f] > 0).float()
        recon  = recon + F_torch.binary_cross_entropy_with_logits(
            decoder_out[:, :, f], target, reduction='mean'
        )
        n_terms += 1

    # ZI-lognorm: gate BCE + MSE on non-zero part (sigmoid already applied)
    zi_idx = feat_type_idx.get('zi_lognorm', [])
    if gate_logit is not None and zi_idx:
        for s_local, s_global in enumerate(zi_idx):
            gate_label = (x[:, :, s_global] > 0).float()
            recon = recon + 0.5 * F_torch.binary_cross_entropy_with_logits(
                gate_logit[:, :, s_local], gate_label, reduction='mean'
            )
            # MSE only where real value is non-zero (ignore structural zeros)
            mask  = gate_label.bool()
            if mask.any():
                recon = recon + 0.5 * F_torch.mse_loss(
                    decoder_out[:, :, s_global][mask],
                    x[:, :, s_global][mask],
                    reduction='mean'
                )
            n_terms += 1

    # Log-normal and continuous: MSE (sigmoid output vs normalised target)
    for type_key in ('lognormal', 'continuous'):
        for f in feat_type_idx.get(type_key, []):
            recon   = recon + F_torch.mse_loss(
                decoder_out[:, :, f], x[:, :, f], reduction='mean'
            )
            n_terms += 1

    recon_total    = loss_factor * recon / max(n_terms, 1)
    kl_elementwise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_per_dim     = kl_elementwise.mean(dim=0)
    kl_loss        = torch.clamp(kl_per_dim, min=free_bits).sum()

    total = recon_total + kl_weight * kl_loss
    return total, (recon / max(n_terms, 1)).item(), kl_loss.item()


def train_zirvae(X_np, window_size, n_features, feat_type_idx,
                 hidden_dim=128, latent_dim=64, n_layers=1,
                 epochs=300, batch_size=256, lr=1e-3,
                 noise_std=0.10, free_bits=1.0, n_cycles=4,
                 device='cpu'):
    X   = X_np.reshape(-1, window_size, n_features).astype(np.float32)
    X_t = torch.tensor(X, device=device)

    model     = ZIRVAE(n_features, window_size, hidden_dim, latent_dim,
                       feat_type_idx, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader    = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True
    )
    loss_factor = float(window_size * n_features)

    # Noise should not be added to Bernoulli features (they are binary)
    bern_idx = feat_type_idx.get('bernoulli', [])

    model.train()
    for epoch in range(epochs):
        kl_weight = _cyclical_kl_weight(epoch, epochs, n_cycles=n_cycles)
        e_loss = e_recon = e_kl = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()
            if noise_std > 0:
                noise = noise_std * torch.randn_like(batch)
                if bern_idx:
                    noise[:, :, bern_idx] = 0.0   # no noise on binary features
                noisy = (batch + noise).clamp(0, 1)
            else:
                noisy = batch
            decoder_out, gate_logit, mu, log_var = model(noisy)
            loss, r, k = zirvae_loss(
                batch, decoder_out, gate_logit, mu, log_var,
                feat_type_idx, kl_weight, loss_factor, free_bits
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            e_loss += loss.item(); e_recon += r; e_kl += k

        n_b = max(len(loader), 1)
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch [{epoch+1:>4}/{epochs}]  "
                  f"loss={e_loss/n_b:.4f}  recon={e_recon/n_b:.4f}  "
                  f"kl={e_kl/n_b:.4f}  kl_w={kl_weight:.2f}")

    model.eval()
    return model


# ═══════════════════════════════════════════════════
# EVALUATION HELPERS
# ═══════════════════════════════════════════════════

def ks_per_class(real_X, real_y, syn_X, syn_y, feat_cols):
    """KS test separately for class 0 and class 1."""
    results = {}
    for cls in [0, 1]:
        r_flat = real_X[real_y == cls].mean(axis=1)   # (N, F)
        s_flat = syn_X[syn_y == cls].mean(axis=1)
        rows   = []
        for i, col in enumerate(feat_cols):
            s, p = stats.ks_2samp(r_flat[:, i], s_flat[:, i])
            rows.append({'feature': col,
                         'ks_stat': round(float(s), 4),
                         'p_value': round(float(p), 4),
                         'similar': bool(p > 0.05)})
        results[cls] = pd.DataFrame(rows).sort_values('ks_stat', ascending=False)
    return results


class LSTMClassifier(nn.Module):
    """Lightweight LSTM for binary classification on (B, T, F) sequences."""
    def __init__(self, n_features, hidden_dim=64, n_layers=1, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers,
                            batch_first=True, dropout=0.0)
        self.fc   = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)   # h: (n_layers, B, hidden)
        return self.fc(h[-1])      # (B, n_classes)


def _lstm_train_eval(X_tr, y_tr, X_te, y_te,
                     epochs=30, batch_size=128, lr=1e-3, device='cpu'):
    """
    Train a small LSTM on (N, T, F) sequences, evaluate on held-out sequences.
    Returns (macro F1, trained clf).
    """
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long,    device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    F   = X_tr.shape[2]
    clf = LSTMClassifier(F, hidden_dim=64, n_layers=1).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True
    )

    clf.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(clf(xb), yb).backward()
            opt.step()

    clf.eval()
    with torch.no_grad():
        preds = clf(X_te_t).argmax(dim=1).cpu().numpy()

    f1 = float(f1_score(y_te, preds, average='macro', zero_division=0))
    return f1, clf


def lstm_score(X_tr, y_tr, X_te, y_te,
               epochs=30, batch_size=128, lr=1e-3, device='cpu'):
    f1, _ = _lstm_train_eval(X_tr, y_tr, X_te, y_te, epochs, batch_size, lr, device)
    return f1


def permutation_importance(clf, X_te, y_te, feat_cols, device='cpu', n_repeats=5):
    """
    For each feature, shuffle its values across all timesteps n_repeats times
    and measure the average drop in macro F1.

    A large drop → feature is important for the LSTM.
    A small/zero drop → feature can be ignored.

    Returns a DataFrame sorted by importance (descending).
    """
    X_te_t   = torch.tensor(X_te, dtype=torch.float32, device=device)
    y_te_np  = y_te

    clf.eval()
    with torch.no_grad():
        base_preds = clf(X_te_t).argmax(dim=1).cpu().numpy()
    base_f1 = float(f1_score(y_te_np, base_preds, average='macro', zero_division=0))

    rng  = np.random.default_rng(seed=0)
    rows = []

    for f_idx, col in enumerate(feat_cols):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_te.copy()
            # shuffle this feature across all samples and timesteps
            flat         = X_perm[:, :, f_idx].reshape(-1)
            flat_shuffled = rng.permutation(flat)
            X_perm[:, :, f_idx] = flat_shuffled.reshape(X_te.shape[0], X_te.shape[1])

            X_perm_t = torch.tensor(X_perm, dtype=torch.float32, device=device)
            with torch.no_grad():
                preds = clf(X_perm_t).argmax(dim=1).cpu().numpy()
            perm_f1 = float(f1_score(y_te_np, preds, average='macro', zero_division=0))
            drops.append(base_f1 - perm_f1)

        mean_drop = float(np.mean(drops))
        rows.append({'feature': col, 'importance': round(mean_drop, 4),
                     'base_f1': round(base_f1, 4)})

    df = pd.DataFrame(rows).sort_values('importance', ascending=False).reset_index(drop=True)
    return df, base_f1


# ═══════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════
print("\n── 1. Loading data ──────────────────────────────")
print(f"   Train files ({len(TRAIN_FILES)}): {sorted(TRAIN_FILES)}")
print(f"   Test  files ({len(TEST_FILES)}):  {sorted(TEST_FILES)}")
df_train = load_files(TRAIN_FILES, DATA_DIR)
df_test  = load_files(TEST_FILES,  DATA_DIR)
print(f"   Train rows: {len(df_train)}  |  Test rows: {len(df_test)}")
print(f"   Train labels: {dict(zip(*np.unique(df_train['label'].values, return_counts=True)))}")
print(f"   Test  labels: {dict(zip(*np.unique(df_test['label'].values,  return_counts=True)))}")

# Detect distribution types BEFORE any transformation
_feat_cols_raw = [c for c in df_train.columns if c != 'label']
_feat_types    = detect_feature_types(df_train, _feat_cols_raw, RANK_COLS)

X_tr, y_tr, X_te, y_te, feat_cols, sparse_idx = preprocess_and_window(
    df_train, df_test, WINDOW_SIZE
)
N, T, F = X_tr.shape

# Build feat_type_idx: map type string → list of feature indices (post-preprocess order)
feat_type_idx = {'bernoulli': [], 'zi_lognorm': [], 'lognormal': [], 'continuous': []}
for col in feat_cols:
    ftype = _feat_types.get(col, 'continuous')
    feat_type_idx[ftype].append(feat_cols.index(col))

print(f"   Windows — Train: {X_tr.shape}  Test: {X_te.shape}")
print(f"   Features: {feat_cols}")
print(f"   Feature types detected:")
for ftype, fidx in feat_type_idx.items():
    if fidx:
        print(f"     {ftype:18s}: {[feat_cols[i] for i in fidx]}")

# flat (mean over T) — used only for correlation heatmap
X_tr_flat = X_tr.mean(axis=1)

# shared hyperparams
latent_dim = max(4, min((T * F) // 10, 32))
hidden_dim = max(128, F * 8)
free_bits  = round(min(0.5, 8.0 / latent_dim), 4)
batch_size = min(256, max(32, N // 20))
print(f"\n   Hyperparams: latent={latent_dim}  hidden={hidden_dim}  "
      f"free_bits={free_bits}  batch={batch_size}")


# ═══════════════════════════════════════════════════
# 2. BASELINE LSTM
# ═══════════════════════════════════════════════════
print("\n── 2. Baseline LSTM ─────────────────────────────")
base_f1, _ = _lstm_train_eval(X_tr, y_tr, X_te, y_te, device=DEVICE)
print(f"   Baseline (real→real) F1 : {base_f1:.4f}")


# ═══════════════════════════════════════════════════
# 3. TRAIN — three architectures, all features
# ═══════════════════════════════════════════════════

# Transformer: d_model must be divisible by n_heads=4
d_model = max(32, (latent_dim // 4) * 4)

print(f"\n   Generator hyperparams: latent={latent_dim}  hidden={hidden_dim}  "
      f"free_bits={free_bits}  d_model={d_model}")


def _train_one_arch(arch_name, X_cls):
    """Train one generator for one class, return sampled (N_SYNTH, T, F)."""
    X_flat = X_cls.reshape(len(X_cls), -1)

    if arch_name == 'ZI-RVAE':
        model = train_zirvae(
            X_np          = X_flat,
            window_size   = T,
            n_features    = F,
            feat_type_idx = feat_type_idx,
            hidden_dim    = hidden_dim,
            latent_dim    = latent_dim,
            n_layers      = 1,
            epochs        = EPOCHS,
            batch_size    = batch_size,
            lr            = 1e-3,
            noise_std     = 0.05,
            free_bits     = free_bits,
            n_cycles      = 4,
            device        = DEVICE,
        )
        return model.sample(N_SYNTH_PER_CLASS, device=DEVICE)

    elif arch_name == 'LSTM-Attn':
        model = train_lstm_attn_vae(
            X_np           = X_flat,
            window_size    = T,
            n_raw_features = F,
            hidden_dim     = hidden_dim,
            latent_dim     = latent_dim,
            n_layers       = 1,
            epochs         = EPOCHS,
            batch_size     = batch_size,
            lr             = 1e-3,
            noise_std      = 0.05,
            free_bits      = free_bits,
            n_cycles       = 4,
            device         = DEVICE,
        )
        return model.sample(N_SYNTH_PER_CLASS, device=DEVICE).numpy()

    else:  # Transformer
        model = train_transformer_vae(
            X_np            = X_flat,
            window_size     = T,
            n_raw_features  = F,
            d_model         = d_model,
            latent_dim      = latent_dim,
            n_heads         = 4,
            n_layers        = 2,
            dim_feedforward = hidden_dim,
            epochs          = EPOCHS,
            batch_size      = batch_size,
            lr              = 1e-3,
            noise_std       = 0.05,
            free_bits       = free_bits,
            n_cycles        = 4,
            device          = DEVICE,
        )
        return model.sample(N_SYNTH_PER_CLASS, device=DEVICE).numpy()


ARCH_NAMES   = ['ZI-RVAE', 'LSTM-Attn', 'Transformer']
arch_results = {}

for arch_name in ARCH_NAMES:
    print(f"\n{'='*55}")
    print(f"  Architecture: {arch_name}")
    print(f"{'='*55}")

    syn_X_parts, syn_y_parts = [], []

    for cls in [0, 1]:
        X_cls = X_tr[y_tr == cls]
        print(f"\n  Class {cls}: {len(X_cls)} windows")
        if len(X_cls) < batch_size:
            print(f"  [skip] too few samples")
            continue
        X_syn = _train_one_arch(arch_name, X_cls)   # (N, T, F)
        syn_X_parts.append(X_syn)
        syn_y_parts.append(np.full(N_SYNTH_PER_CLASS, cls, dtype=np.int64))

    X_syn_all = np.concatenate(syn_X_parts, axis=0)
    y_syn_all = np.concatenate(syn_y_parts, axis=0)

    ks_arch = ks_per_class(X_tr, y_tr, X_syn_all, y_syn_all, feat_cols)
    tstr    = lstm_score(X_syn_all, y_syn_all, X_te, y_te, device=DEVICE)
    trts    = lstm_score(X_tr,      y_tr,      X_syn_all, y_syn_all, device=DEVICE)

    arch_results[arch_name] = {
        'X_syn_all'  : X_syn_all,
        'y_syn_all'  : y_syn_all,
        'ks_by_class': ks_arch,
        'tstr_f1'    : tstr,
        'trts_f1'    : trts,
    }

    for cls in [0, 1]:
        ks_df = ks_arch[cls]
        label = 'Normal' if cls == 0 else 'Attack'
        print(f"  [{arch_name}] Class {cls} ({label}): "
              f"KS pass {ks_df['similar'].sum()}/{len(ks_df)}  "
              f"mean_KS={ks_df['ks_stat'].mean():.4f}")
    print(f"  [{arch_name}] TSTR={tstr:.4f}  TRTS={trts:.4f}  Baseline={base_f1:.4f}")


# ═══════════════════════════════════════════════════
# 4. PLOTS — per architecture
# ═══════════════════════════════════════════════════
print("\n── 4. Saving plots ──────────────────────────────")

arch_colors = {'ZI-RVAE': 'darkorange', 'LSTM-Attn': 'seagreen', 'Transformer': 'mediumpurple'}
real_df     = pd.DataFrame(X_tr_flat, columns=feat_cols)

for arch_name in ARCH_NAMES:
    res        = arch_results[arch_name]
    X_syn_arch = res['X_syn_all']
    y_syn_arch = res['y_syn_all']
    ks_arch    = res['ks_by_class']
    color      = arch_colors[arch_name]
    safe_name  = arch_name.replace(' ', '_').replace('-', '')

    # ── KS bar chart ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)
    for ax, cls in zip(axes, [0, 1]):
        ks_df      = ks_arch[cls].sort_values('ks_stat', ascending=False)
        bar_colors = ['steelblue' if r else 'tomato' for r in ks_df['similar']]
        ax.bar(ks_df['feature'], ks_df['ks_stat'], color=bar_colors)
        ax.axhline(0.05, color='black', linestyle='--', lw=1.2)
        label = 'Normal' if cls == 0 else 'Attack'
        ax.set_title(f'Class {cls} ({label})  mean KS={ks_df["ks_stat"].mean():.3f}  '
                     f'pass={ks_df["similar"].sum()}/{len(ks_df)}')
        ax.set_ylabel('KS Statistic'); ax.set_ylim(0, 1.05)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    fig.suptitle(f'KS per feature — {arch_name}  (blue=pass, red=fail)', fontsize=11)
    path = os.path.join(OUT_DIR, f'ks_{safe_name}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    print(f"   Saved: {path}")

    # ── Distribution overlays ─────────────────────
    fig, axes = plt.subplots(2, F, figsize=(2.8 * F, 3.5 * 2), constrained_layout=True)
    for row, cls in enumerate([0, 1]):
        r_flat = X_te[y_te == cls].reshape(-1, F)
        s_flat = X_syn_arch[y_syn_arch == cls].reshape(-1, F)
        ks_df  = ks_arch[cls].set_index('feature')
        label  = 'Normal' if cls == 0 else 'Attack'
        for col_idx, col in enumerate(feat_cols):
            ax = axes[row, col_idx]
            r  = r_flat[:, col_idx]
            s  = s_flat[:, col_idx]
            ax.hist(r, bins=25, alpha=0.35, color='steelblue', density=True, label='Real')
            ax.hist(s, bins=25, alpha=0.35, color=color,       density=True, label=arch_name)
            try:
                xs = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 150)
                ax.plot(xs, stats.gaussian_kde(r)(xs), color='steelblue', lw=1.2)
                ax.plot(xs, stats.gaussian_kde(s)(xs), color=color, lw=1.2)
            except Exception:
                pass
            ks_v = ks_df.loc[col, 'ks_stat'] if col in ks_df.index else float('nan')
            ax.set_title(f"{col}\nKS={ks_v:.3f}", fontsize=6)
            ax.tick_params(labelsize=5)
            if col_idx == 0:
                ax.set_ylabel(f'Class {cls}\n({label})', fontsize=7)
            if row == 0 and col_idx == 0:
                ax.legend(fontsize=5)
    fig.suptitle(f'Distribution Overlay — {arch_name}\nRow 0: Normal  Row 1: Attack', fontsize=10)
    path = os.path.join(OUT_DIR, f'distributions_{safe_name}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    print(f"   Saved: {path}")

    # ── Correlation heatmap ───────────────────────
    syn_df = pd.DataFrame(X_syn_arch.mean(axis=1), columns=feat_cols)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    rc = real_df.corr(); sc = syn_df.corr(); dc = (rc - sc).abs()
    kw = dict(cmap='coolwarm', vmin=-1, vmax=1, square=True,
              annot=True, fmt='.2f', annot_kws={'size': 6})
    sns.heatmap(rc, ax=a1, **kw);  a1.set_title('Real')
    sns.heatmap(sc, ax=a2, **kw);  a2.set_title(f'Synthetic ({arch_name})')
    sns.heatmap(dc, ax=a3, cmap='Reds', vmin=0, vmax=1, square=True,
                annot=True, fmt='.2f', annot_kws={'size': 6})
    a3.set_title('|Difference|')
    fig.suptitle(f'Correlation — {arch_name}', fontsize=11)
    path = os.path.join(OUT_DIR, f'correlations_{safe_name}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    print(f"   Saved: {path}")

# ── TSTR / TRTS comparison bar chart ─────────────
fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
x      = np.arange(3)
width  = 0.22
labels = ['Baseline\n(real→real)', 'TSTR\n(syn→real)', 'TRTS\n(real→syn)']
for i, arch_name in enumerate(ARCH_NAMES):
    res  = arch_results[arch_name]
    vals = [base_f1, res['tstr_f1'], res['trts_f1']]
    off  = (i - 1) * width
    bars = ax.bar(x + off, vals, width, label=arch_name,
                  color=arch_colors[arch_name], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15); ax.set_ylabel('Macro F1')
ax.axhline(0.5, color='red', linestyle='--', lw=1, label='random')
ax.legend(); ax.set_title('TSTR / TRTS / Baseline — Architecture Comparison')
path = os.path.join(OUT_DIR, 'tstr_trts_comparison.png')
plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
print(f"   Saved: {path}")


# ═══════════════════════════════════════════════════
# 5. SUMMARY
# ═══════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════")
print("  SUMMARY — Architecture Comparison")
print(f"  Train: {sorted(TRAIN_FILES)}")
print(f"  Test : {sorted(TEST_FILES)}")
print("══════════════════════════════════════════════════")
header = f"  {'Metric':<30} {'ZI-RVAE':>12} {'LSTM-Attn':>12} {'Transformer':>12}"
print(header); print('-' * len(header))
for cls in [0, 1]:
    label    = 'Normal' if cls == 0 else 'Attack'
    row_ks   = [f"{arch_results[a]['ks_by_class'][cls]['ks_stat'].mean():.4f}" for a in ARCH_NAMES]
    row_pass = [f"{arch_results[a]['ks_by_class'][cls]['similar'].sum()}"
                f"/{len(arch_results[a]['ks_by_class'][cls])}" for a in ARCH_NAMES]
    print(f"  {'KS mean  cls'+str(cls)+' ('+label+')':<30} {row_ks[0]:>12} {row_ks[1]:>12} {row_ks[2]:>12}")
    print(f"  {'KS pass  cls'+str(cls)+' ('+label+')':<30} {row_pass[0]:>12} {row_pass[1]:>12} {row_pass[2]:>12}")
tstr_vals = [f"{arch_results[a]['tstr_f1']:.4f}" for a in ARCH_NAMES]
trts_vals = [f"{arch_results[a]['trts_f1']:.4f}" for a in ARCH_NAMES]
base_str  = f"{base_f1:.4f}"
print(f"  {'Baseline F1 (real→real)':<30} {base_str:>12} {base_str:>12} {base_str:>12}")
print(f"  {'TSTR F1  (syn→real)':<30} {tstr_vals[0]:>12} {tstr_vals[1]:>12} {tstr_vals[2]:>12}")
print(f"  {'TRTS F1  (real→syn)':<30} {trts_vals[0]:>12} {trts_vals[1]:>12} {trts_vals[2]:>12}")
print(f"\n  Results saved to: {OUT_DIR}")
print("══════════════════════════════════════════════════\n")
