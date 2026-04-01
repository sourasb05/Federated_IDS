# feature_select_zi_compare.py
#
# Stage 1 — Feature Selection (across all 20 files, 16 used for selection)
#   Filter methods:
#     - ANOVA F-statistic  (univariate separability between classes)
#     - Mutual Information (non-linear univariate relevance)
#     - Spearman |rho|     (monotonic correlation with label)
#   Embedded method:
#     - Random Forest Gini importance (after windowing + preprocessing)
#   Importance scores are normalised to [0,1] per method, then averaged
#   across all 16 files AND across the 4 methods → one final score per feature.
#   A threshold (top-k or score cut-off) selects the feature subset.
#
# Stage 2 — Generative Models on Selected Features
#   All three architectures now have Zero-Inflation decoders:
#     ZI-RVAE           (GRU encoder + ZI decoder)        — from zirvae_multifile.py
#     ZI-LSTM-Attn      (LSTM+Attention encoder + ZI decoder)
#     ZI-Transformer    (Transformer encoder + ZI decoder; MLP body same as before)
#
#   ZI-decoder is identical to ZIRVAEDecoder: Gaussian head for dense features,
#   Bernoulli gate head for sparse features (>30% zeros after preprocessing).
#
# Stage 3 — Evaluation (same as zirvae_multifile.py)
#   KS test per class, TSTR, TRTS, Baseline F1, distribution plots,
#   correlation heatmaps, summary table.
#
# Run:  python toy/feature_select_zi_compare.py

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F_torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)

from rvae import REncoder   # reused for all three encoders

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'attack_data', 'blackhole_var20_base'
)
_all_files = [f'{i}_features_timeseries_60_sec.csv' for i in range(1, 21)]
_rng       = np.random.default_rng(seed=42)
_shuffled  = _rng.permutation(_all_files).tolist()

# 16 files for feature selection; same train/test split for model training
FS_FILES    = _shuffled[:16]      # 16 files for feature importance aggregation
TRAIN_FILES = _shuffled[:14]     # 14 for model training  (subset of FS_FILES)
TEST_FILES  = _shuffled[14:]     # 6  for evaluation

WINDOW_SIZE       = 10
N_SYNTH_PER_CLASS = 1000
EPOCHS            = 1000
DEVICE            = 'cpu'
SPARSE_THRESH     = 0.30
RANK_COLS         = ['rank', 'rank.1']
TOP_K_FEATURES    = 8   # number of features to keep after selection

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'feature_select_zi_results')
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
# DATA LOADING + PREPROCESSING (reused from zirvae_multifile)
# ═══════════════════════════════════════════════════

def load_files(file_list, data_dir):
    dfs = []
    for fname in file_list:
        path = os.path.join(data_dir, fname)
        df   = pd.read_csv(path, encoding='utf-8', encoding_errors='ignore')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def preprocess_and_window(df_tr, df_te, window_size, feat_cols=None):
    """
    Preprocessing identical to zirvae_multifile.preprocess_and_window.
    If feat_cols is provided, only those columns are used (after engineering).
    Returns X_tr, y_tr, X_te, y_te, feat_cols_out, sparse_idx
    """
    all_feat_cols = [c for c in df_tr.columns if c != 'label']

    # 1. Differencing + sign-log1p for rank columns
    rank_cols_present = [c for c in RANK_COLS if c in all_feat_cols]
    for df in [df_tr, df_te]:
        if rank_cols_present:
            diff = df[rank_cols_present].diff().fillna(0)
            df[rank_cols_present] = np.sign(diff) * np.log1p(np.abs(diff))

    # 2. Sparse detection on train
    sparse_candidates = [
        c for c in all_feat_cols
        if (df_tr[c] == 0).mean() > SPARSE_THRESH
    ]

    # 3. log1p on sparse cols
    for df in [df_tr, df_te]:
        if sparse_candidates:
            df[sparse_candidates] = np.log1p(df[sparse_candidates])

    # 4. Min-max normalisation fitted on train
    g_min = df_tr[all_feat_cols].min()
    g_max = df_tr[all_feat_cols].max()
    denom = (g_max - g_min).replace(0, 1)
    for df in [df_tr, df_te]:
        df[all_feat_cols] = ((df[all_feat_cols] - g_min) / denom).clip(0, 1).fillna(0)

    # 5. Symmetric rescaling for rank cols
    for col in rank_cols_present:
        abs_max = df_tr[col].abs().max()
        if abs_max > 0:
            for df in [df_tr, df_te]:
                df[col] = (df[col] / (2 * abs_max) + 0.5).clip(0, 1)

    # Select feature subset if provided
    use_cols = feat_cols if feat_cols is not None else all_feat_cols
    use_cols = [c for c in use_cols if c in df_tr.columns]  # safety

    # Sparse idx within the selected feature list
    sparse_idx = [use_cols.index(c) for c in sparse_candidates if c in use_cols]

    # 6. Sliding windows
    def _windows(df):
        X, y   = [], []
        vals   = df[use_cols].values.astype(np.float32)
        labels = df['label'].values.astype(int)
        for i in range(len(vals) - window_size):
            lbl = 1 if 1 in labels[i:i + window_size] else 0
            X.append(vals[i:i + window_size])
            y.append(lbl)
        return np.array(X, np.float32), np.array(y, np.int64)

    X_tr, y_tr = _windows(df_tr)
    X_te, y_te = _windows(df_te)
    return X_tr, y_tr, X_te, y_te, use_cols, sparse_idx


# ═══════════════════════════════════════════════════
# STAGE 1 — FEATURE SELECTION
# ═══════════════════════════════════════════════════

def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise a 1-D score array to [0, 1]. Ties stay equal."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.ones_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def feature_importance_one_file(csv_path: str):
    """
    Compute per-feature importance from a single CSV using four methods.

    The file is loaded raw (no windowing) so the label column aligns row-by-row.
    Rank columns are differenced first (same as preprocessing pipeline).

    Returns:
        feat_cols : list[str]
        scores    : dict { method_name -> np.ndarray(F,) normalised }
    """
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    feat_cols = [c for c in df.columns if c != 'label']

    # Differencing for rank cols
    rank_cols_present = [c for c in RANK_COLS if c in feat_cols]
    if rank_cols_present:
        diff = df[rank_cols_present].diff().fillna(0)
        df[rank_cols_present] = np.sign(diff) * np.log1p(np.abs(diff))

    # log1p on sparse cols
    sparse_cols = [c for c in feat_cols if (df[c] == 0).mean() > SPARSE_THRESH]
    if sparse_cols:
        df[sparse_cols] = np.log1p(df[sparse_cols])

    # Min-max
    g_min = df[feat_cols].min()
    g_max = df[feat_cols].max()
    denom = (g_max - g_min).replace(0, 1)
    df[feat_cols] = ((df[feat_cols] - g_min) / denom).clip(0, 1).fillna(0)

    # Symmetric rescaling for rank
    for col in rank_cols_present:
        abs_max = df[col].abs().max()
        if abs_max > 0:
            df[col] = (df[col] / (2 * abs_max) + 0.5).clip(0, 1)

    X = df[feat_cols].values
    y = df['label'].values.astype(int)

    if len(np.unique(y)) < 2:
        return feat_cols, None   # skip single-class files

    scores = {}

    # ── Filter: ANOVA F-statistic ─────────────────
    f_vals, _ = f_classif(X, y)
    f_vals     = np.nan_to_num(f_vals, nan=0.0, posinf=0.0)
    scores['anova_f'] = _normalise_scores(f_vals)

    # ── Filter: Mutual Information ────────────────
    mi = mutual_info_classif(X, y, random_state=0)
    scores['mutual_info'] = _normalise_scores(mi)

    # ── Filter: Spearman |rho| ────────────────────
    rho = np.array([
        abs(stats.spearmanr(X[:, i], y).statistic)
        if len(np.unique(X[:, i])) > 1 else 0.0
        for i in range(X.shape[1])
    ])
    scores['spearman_rho'] = _normalise_scores(rho)

    # ── Embedded: Random Forest Gini ─────────────
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6,
        class_weight='balanced', random_state=0, n_jobs=-1
    )
    rf.fit(X, y)
    scores['rf_gini'] = _normalise_scores(rf.feature_importances_)

    return feat_cols, scores


def aggregate_feature_importance(fs_files, data_dir):
    """
    Run feature_importance_one_file() on each FS file.
    Average normalised scores across files and methods.

    Returns DataFrame with columns:
        feature | anova_f | mutual_info | spearman_rho | rf_gini | mean_score | rank
    sorted by mean_score descending.
    """
    print(f"\n── Stage 1: Feature Selection across {len(fs_files)} files ──")

    all_methods = ['anova_f', 'mutual_info', 'spearman_rho', 'rf_gini']
    # accumulator: method -> list of score arrays
    accum  = {m: [] for m in all_methods}
    feat_cols_ref = None
    n_valid = 0

    for fname in fs_files:
        path = os.path.join(data_dir, fname)
        feat_cols, sc = feature_importance_one_file(path)

        if feat_cols_ref is None:
            feat_cols_ref = feat_cols

        if sc is None:
            print(f"   [skip] {fname} — single-class file")
            continue

        n_valid += 1
        for m in all_methods:
            accum[m].append(sc[m])
        print(f"   [{n_valid:>2}] {fname}  RF_top={feat_cols[sc['rf_gini'].argmax()]}")

    print(f"\n   Valid files: {n_valid}/{len(fs_files)}")

    # Average across files
    avg = {m: np.mean(accum[m], axis=0) for m in all_methods}

    # Average across methods → final score
    mean_score = np.mean([avg[m] for m in all_methods], axis=0)

    rows = []
    for i, col in enumerate(feat_cols_ref):
        rows.append({
            'feature'      : col,
            'anova_f'      : round(float(avg['anova_f'][i]),      4),
            'mutual_info'  : round(float(avg['mutual_info'][i]),  4),
            'spearman_rho' : round(float(avg['spearman_rho'][i]), 4),
            'rf_gini'      : round(float(avg['rf_gini'][i]),      4),
            'mean_score'   : round(float(mean_score[i]),          4),
        })

    df_imp = pd.DataFrame(rows).sort_values('mean_score', ascending=False)
    df_imp['rank'] = range(1, len(df_imp) + 1)
    return df_imp


# ═══════════════════════════════════════════════════
# STAGE 2 — ZI ARCHITECTURES
# ═══════════════════════════════════════════════════
# All three share the same ZIDecoder — only the encoder differs.
# ZIDecoder is a GRU body (identical to ZIRVAEDecoder in zirvae_multifile.py).

class ZIDecoder(nn.Module):
    """
    Shared Zero-Inflated GRU decoder for all three ZI architectures.

    Dense features   → Gaussian (sigmoid) head.
    Sparse features  → Bernoulli gate (logit) head.

    Synthesis: out = gate * gauss_val  (exact zeros for closed gates).
    """

    def __init__(self, latent_dim, hidden_dim, n_features,
                 window_size, sparse_idx, n_layers=1):
        super().__init__()
        self.window_size = window_size
        self.n_layers    = n_layers
        self.hidden_dim  = hidden_dim
        self.sparse_idx  = sparse_idx

        self.fc_h0    = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_input = nn.Linear(latent_dim, n_features)
        self.gru      = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_features)
        self.gate_fc  = nn.Linear(hidden_dim, len(sparse_idx)) if sparse_idx else None

    def forward(self, z):
        B   = z.size(0)
        h0  = self.fc_h0(z).view(self.n_layers, B, self.hidden_dim)
        inp = self.fc_input(z).unsqueeze(1).expand(-1, self.window_size, -1)
        gru_out, _ = self.gru(inp, h0)
        gauss_out  = torch.sigmoid(self.fc_out(gru_out))
        gate_logit = self.gate_fc(gru_out) if self.gate_fc is not None else None
        return gauss_out, gate_logit

    @torch.no_grad()
    def sample(self, z):
        gauss_out, gate_logit = self.forward(z)
        out = gauss_out.clone()
        if gate_logit is not None:
            gate_prob = torch.sigmoid(gate_logit)
            gate      = torch.bernoulli(gate_prob).bool()
            for s_local, s_global in enumerate(self.sparse_idx):
                out[:, :, s_global] = torch.where(
                    gate[:, :, s_local],
                    gauss_out[:, :, s_global],
                    torch.zeros_like(gauss_out[:, :, s_global])
                )
        return out


# ──────────────────────────────────────────────────
# ZI-RVAE: GRU encoder (REncoder from rvae.py) + ZIDecoder
# ──────────────────────────────────────────────────

class ZI_RVAE(nn.Module):
    def __init__(self, n_features, window_size, hidden_dim,
                 latent_dim, sparse_idx, n_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = REncoder(n_features, hidden_dim, latent_dim, n_layers)
        self.decoder    = ZIDecoder(latent_dim, hidden_dim, n_features,
                                    window_size, sparse_idx, n_layers)

    def forward(self, x):
        mu, log_var           = self.encoder(x)
        z                     = self.encoder.reparameterize(mu, log_var)
        gauss_out, gate_logit = self.decoder(z)
        return gauss_out, gate_logit, mu, log_var

    @torch.no_grad()
    def sample(self, n, device='cpu'):
        z   = torch.randn(n, self.latent_dim, device=device)
        out = self.decoder.sample(z)
        return out.cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────────
# ZI-LSTM-Attn: LSTM+Attention encoder + ZIDecoder
# ──────────────────────────────────────────────────

class LSTMAttnEncoder_ZI(nn.Module):
    """LSTM encoder with dot-product attention over all T timesteps."""

    def __init__(self, n_features, hidden_dim, latent_dim, n_layers=1):
        super().__init__()
        self.lstm   = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.fc_var.bias, -1.0)

    def forward(self, x):
        out, _ = self.lstm(x)                           # (B, T, H)
        weights = torch.softmax(self.attn_w(out), dim=1)  # (B, T, 1)
        context = (weights * out).sum(dim=1)             # (B, H)
        return self.fc_mu(context), self.fc_var(context)

    @staticmethod
    def reparameterize(mu, log_var):
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)


class ZI_LSTMAttn(nn.Module):
    def __init__(self, n_features, window_size, hidden_dim,
                 latent_dim, sparse_idx, n_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = LSTMAttnEncoder_ZI(n_features, hidden_dim, latent_dim, n_layers)
        self.decoder    = ZIDecoder(latent_dim, hidden_dim, n_features,
                                    window_size, sparse_idx, n_layers)

    def forward(self, x):
        mu, log_var           = self.encoder(x)
        z                     = LSTMAttnEncoder_ZI.reparameterize(mu, log_var)
        gauss_out, gate_logit = self.decoder(z)
        return gauss_out, gate_logit, mu, log_var

    @torch.no_grad()
    def sample(self, n, device='cpu'):
        z   = torch.randn(n, self.latent_dim, device=device)
        out = self.decoder.sample(z)
        return out.cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────────
# ZI-Transformer: Transformer encoder + ZIDecoder
# ──────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerEncoder_ZI(nn.Module):
    """Transformer encoder: mean-pool over T → mu, log_var."""

    def __init__(self, n_features, d_model, latent_dim,
                 n_heads=4, n_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc_mu  = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)
        nn.init.constant_(self.fc_var.bias, -1.0)

    def forward(self, x):
        e      = self.pos_enc(self.input_proj(x))
        e      = self.transformer(e)
        pooled = e.mean(dim=1)
        return self.fc_mu(pooled), self.fc_var(pooled)

    @staticmethod
    def reparameterize(mu, log_var):
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)


class ZI_Transformer(nn.Module):
    def __init__(self, n_features, window_size, hidden_dim, latent_dim,
                 sparse_idx, d_model=64, n_heads=4, n_layers=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = TransformerEncoder_ZI(
            n_features, d_model, latent_dim,
            n_heads, n_layers, dim_feedforward, dropout
        )
        self.decoder    = ZIDecoder(latent_dim, hidden_dim, n_features,
                                    window_size, sparse_idx)

    def forward(self, x):
        mu, log_var           = self.encoder(x)
        z                     = TransformerEncoder_ZI.reparameterize(mu, log_var)
        gauss_out, gate_logit = self.decoder(z)
        return gauss_out, gate_logit, mu, log_var

    @torch.no_grad()
    def sample(self, n, device='cpu'):
        z   = torch.randn(n, self.latent_dim, device=device)
        out = self.decoder.sample(z)
        return out.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════
# SHARED ZI LOSS + CYCLICAL KL
# ═══════════════════════════════════════════════════

def _cyclical_kl_weight(epoch, n_epochs, n_cycles=4, min_w=0.0, max_w=1.0):
    cycle_len = max(n_epochs / n_cycles, 1)
    pos       = (epoch % cycle_len) / cycle_len
    ramp      = min(1.0, pos * 2.0)
    return min_w + (max_w - min_w) * ramp


def zi_loss(x, gauss_out, gate_logit, mu, log_var,
            sparse_idx, kl_weight, loss_factor, free_bits):
    recon_mse = F_torch.mse_loss(gauss_out, x, reduction='mean')

    recon_bce = torch.tensor(0.0, device=x.device)
    if gate_logit is not None and len(sparse_idx) > 0:
        for s_local, s_global in enumerate(sparse_idx):
            gate_label = (x[:, :, s_global] > 0).float()
            recon_bce  = recon_bce + F_torch.binary_cross_entropy_with_logits(
                gate_logit[:, :, s_local], gate_label, reduction='mean'
            )

    recon_total    = loss_factor * recon_mse + 0.1 * recon_bce
    kl_elementwise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_per_dim     = kl_elementwise.mean(dim=0)
    kl_loss        = torch.clamp(kl_per_dim, min=free_bits).sum()
    total          = recon_total + kl_weight * kl_loss
    return total, recon_mse.item(), kl_loss.item()


def train_zi_model(model, X_np, window_size, n_features,
                   sparse_idx, epochs, batch_size, lr,
                   noise_std, free_bits, n_cycles, device, tag=''):
    """Generic training loop for any ZI-* model."""
    X   = X_np.reshape(-1, window_size, n_features).astype(np.float32)
    X_t = torch.tensor(X, device=device)

    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader      = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True
    )
    loss_factor = float(window_size * n_features)

    model.train()
    for epoch in range(epochs):
        kl_w = _cyclical_kl_weight(epoch, epochs, n_cycles=n_cycles)
        e_loss = e_recon = e_kl = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()
            noisy = (batch + noise_std * torch.randn_like(batch)).clamp(0, 1) \
                    if noise_std > 0 else batch
            gauss_out, gate_logit, mu, log_var = model(noisy)
            loss, r, k = zi_loss(
                batch, gauss_out, gate_logit, mu, log_var,
                sparse_idx, kl_w, loss_factor, free_bits
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            e_loss += loss.item(); e_recon += r; e_kl += k

        n_b = max(len(loader), 1)
        if (epoch + 1) % 100 == 0:
            print(f"    [{tag}] Epoch [{epoch+1:>4}/{epochs}]  "
                  f"loss={e_loss/n_b:.4f}  recon={e_recon/n_b:.4f}  "
                  f"kl={e_kl/n_b:.4f}  kl_w={kl_w:.2f}")

    model.eval()
    return model


# ═══════════════════════════════════════════════════
# EVALUATION HELPERS
# ═══════════════════════════════════════════════════

def ks_per_class(real_X, real_y, syn_X, syn_y, feat_cols):
    results = {}
    for cls in [0, 1]:
        r_flat = real_X[real_y == cls].mean(axis=1)
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
    def __init__(self, n_features, hidden_dim=64, n_layers=1, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def _lstm_train_eval(X_tr, y_tr, X_te, y_te,
                     epochs=30, batch_size=128, lr=1e-3, device='cpu'):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long,    device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    F   = X_tr.shape[2]
    clf = LSTMClassifier(F).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True
    )
    clf.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); crit(clf(xb), yb).backward(); opt.step()

    clf.eval()
    with torch.no_grad():
        preds = clf(X_te_t).argmax(dim=1).cpu().numpy()
    return float(f1_score(y_te, preds, average='macro', zero_division=0))


def lstm_score(X_tr, y_tr, X_te, y_te, device='cpu'):
    return _lstm_train_eval(X_tr, y_tr, X_te, y_te, device=device)


# ═══════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════

# ── Stage 1: Feature selection ─────────────────────
df_imp = aggregate_feature_importance(FS_FILES, DATA_DIR)

print("\n  Feature Importance Scores (averaged over 16 files × 4 methods):")
print(df_imp.to_string(index=False))

# Save importance table
imp_path = os.path.join(OUT_DIR, 'feature_importance.csv')
df_imp.to_csv(imp_path, index=False)
print(f"\n  Saved: {imp_path}")

# Plot importance bar chart per method + mean
fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)
methods_plot = ['anova_f', 'mutual_info', 'spearman_rho', 'rf_gini', 'mean_score']
method_titles = ['ANOVA F', 'Mutual Info', 'Spearman |ρ|', 'RF Gini', 'Mean Score']
colors_plot   = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple', 'crimson']

df_plot = df_imp.sort_values('mean_score', ascending=False)
for ax, m, title, col in zip(axes, methods_plot, method_titles, colors_plot):
    ax.barh(df_plot['feature'], df_plot[m], color=col, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.tick_params(axis='y', labelsize=7)
    ax.invert_yaxis()
    if m == 'mean_score':
        # mark selection threshold
        if len(df_plot) >= TOP_K_FEATURES:
            threshold = df_plot['mean_score'].iloc[TOP_K_FEATURES - 1]
            ax.axvline(threshold, color='black', linestyle='--', lw=1.2,
                       label=f'top-{TOP_K_FEATURES} cut')
            ax.legend(fontsize=7)

fig.suptitle(
    f'Feature Importance — averaged over {len(FS_FILES)} files\n'
    f'(top-{TOP_K_FEATURES} selected: shaded)',
    fontsize=10
)
plt.savefig(os.path.join(OUT_DIR, 'feature_importance.png'), dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {os.path.join(OUT_DIR, 'feature_importance.png')}")

# Select top-K features
selected_features = df_imp.head(TOP_K_FEATURES)['feature'].tolist()
print(f"\n  Selected top-{TOP_K_FEATURES} features:")
for rank_i, feat in enumerate(selected_features, 1):
    score = df_imp[df_imp['feature'] == feat]['mean_score'].values[0]
    print(f"    {rank_i:>2}. {feat:<15}  mean_score={score:.4f}")

# ── Load & preprocess data with selected features ──
print("\n── Loading + windowing with selected features ───")
df_train = load_files(TRAIN_FILES, DATA_DIR)
df_test  = load_files(TEST_FILES,  DATA_DIR)

X_tr, y_tr, X_te, y_te, feat_cols, sparse_idx = preprocess_and_window(
    df_train, df_test, WINDOW_SIZE, feat_cols=selected_features
)
N, T, F = X_tr.shape
print(f"  Windows — Train: {X_tr.shape}  Test: {X_te.shape}")
print(f"  Features used: {feat_cols}")
print(f"  Sparse idx: {sparse_idx}  → {[feat_cols[i] for i in sparse_idx]}")

# Shared hyperparams
latent_dim = max(4, min((T * F) // 10, 32))
hidden_dim = max(64, F * 8)
free_bits  = round(min(0.5, 8.0 / latent_dim), 4)
batch_size = min(256, max(32, N // 20))
d_model    = max(32, (latent_dim // 4) * 4)
print(f"\n  Hyperparams: latent={latent_dim}  hidden={hidden_dim}  "
      f"free_bits={free_bits}  batch={batch_size}  d_model={d_model}")

# ── Baseline LSTM ──────────────────────────────────
print("\n── Baseline LSTM (real→real) ─────────────────────")
base_f1 = lstm_score(X_tr, y_tr, X_te, y_te, device=DEVICE)
print(f"  Baseline F1: {base_f1:.4f}")

# ── Stage 2: Train ZI models ───────────────────────
ARCH_DEFS = {
    'ZI-RVAE': lambda: ZI_RVAE(
        F, T, hidden_dim, latent_dim, sparse_idx
    ),
    'ZI-LSTM-Attn': lambda: ZI_LSTMAttn(
        F, T, hidden_dim, latent_dim, sparse_idx
    ),
    'ZI-Transformer': lambda: ZI_Transformer(
        F, T, hidden_dim, latent_dim, sparse_idx,
        d_model=d_model, n_heads=4, n_layers=2,
        dim_feedforward=hidden_dim, dropout=0.1
    ),
}

arch_results = {}

for arch_name, model_fn in ARCH_DEFS.items():
    print(f"\n{'='*58}")
    print(f"  Architecture: {arch_name}")
    print(f"{'='*58}")

    syn_X_parts, syn_y_parts = [], []

    for cls in [0, 1]:
        X_cls = X_tr[y_tr == cls]
        label = 'Normal' if cls == 0 else 'Attack'
        print(f"\n  Class {cls} ({label}): {len(X_cls)} windows")
        if len(X_cls) < batch_size:
            print(f"  [skip] too few samples ({len(X_cls)} < {batch_size})")
            continue

        model = model_fn().to(DEVICE)
        X_flat = X_cls.reshape(len(X_cls), -1)
        model = train_zi_model(
            model      = model,
            X_np       = X_flat,
            window_size = T,
            n_features  = F,
            sparse_idx  = sparse_idx,
            epochs      = EPOCHS,
            batch_size  = batch_size,
            lr          = 1e-3,
            noise_std   = 0.05,
            free_bits   = free_bits,
            n_cycles    = 4,
            device      = DEVICE,
            tag         = f"{arch_name} cls{cls}",
        )

        X_syn = model.sample(N_SYNTH_PER_CLASS, device=DEVICE)   # (N, T, F)
        syn_X_parts.append(X_syn)
        syn_y_parts.append(np.full(N_SYNTH_PER_CLASS, cls, dtype=np.int64))

    if not syn_X_parts:
        print(f"  [skip] no valid classes for {arch_name}")
        continue

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
        label = 'Normal' if cls == 0 else 'Attack'
        ks_df = ks_arch[cls]
        print(f"  [{arch_name}] cls{cls} ({label}): "
              f"KS pass {ks_df['similar'].sum()}/{len(ks_df)}  "
              f"mean_KS={ks_df['ks_stat'].mean():.4f}")
    print(f"  [{arch_name}] TSTR={tstr:.4f}  TRTS={trts:.4f}  Baseline={base_f1:.4f}")


# ── Stage 3: Plots ─────────────────────────────────
print("\n── Stage 3: Saving plots ─────────────────────────")

arch_colors = {
    'ZI-RVAE'       : 'darkorange',
    'ZI-LSTM-Attn'  : 'seagreen',
    'ZI-Transformer': 'mediumpurple',
}
X_tr_flat = X_tr.mean(axis=1)
real_df   = pd.DataFrame(X_tr_flat, columns=feat_cols)

for arch_name, res in arch_results.items():
    X_syn_arch = res['X_syn_all']
    y_syn_arch = res['y_syn_all']
    ks_arch    = res['ks_by_class']
    color      = arch_colors[arch_name]
    safe_name  = arch_name.replace(' ', '_').replace('-', '')

    # KS bar chart
    fig, axes = plt.subplots(1, 2, figsize=(max(10, F * 0.9), 4), constrained_layout=True)
    for ax, cls in zip(axes, [0, 1]):
        ks_df      = ks_arch[cls].sort_values('ks_stat', ascending=False)
        bar_colors = ['steelblue' if r else 'tomato' for r in ks_df['similar']]
        ax.bar(ks_df['feature'], ks_df['ks_stat'], color=bar_colors)
        ax.axhline(0.05, color='black', linestyle='--', lw=1.2)
        label = 'Normal' if cls == 0 else 'Attack'
        ax.set_title(f'cls{cls} ({label})  mean KS={ks_df["ks_stat"].mean():.3f}  '
                     f'pass={ks_df["similar"].sum()}/{len(ks_df)}')
        ax.set_ylabel('KS Statistic'); ax.set_ylim(0, 1.05)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    fig.suptitle(f'KS per feature — {arch_name}  (blue=pass, red=fail)', fontsize=11)
    path = os.path.join(OUT_DIR, f'ks_{safe_name}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

    # Distribution overlays
    fig, axes = plt.subplots(2, F, figsize=(2.8 * F, 3.5 * 2), constrained_layout=True)
    if F == 1:
        axes = axes.reshape(2, 1)
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
                ax.set_ylabel(f'cls{cls} ({label})', fontsize=7)
            if row == 0 and col_idx == 0:
                ax.legend(fontsize=5)
    fig.suptitle(f'Distribution Overlay — {arch_name}', fontsize=10)
    path = os.path.join(OUT_DIR, f'distributions_{safe_name}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

    # Correlation heatmap
    syn_df = pd.DataFrame(X_syn_arch.mean(axis=1), columns=feat_cols)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    rc = real_df.corr(); sc = syn_df.corr(); dc = (rc - sc).abs()
    kw = dict(cmap='coolwarm', vmin=-1, vmax=1, square=True,
              annot=True, fmt='.2f', annot_kws={'size': 7})
    sns.heatmap(rc, ax=a1, **kw);  a1.set_title('Real')
    sns.heatmap(sc, ax=a2, **kw);  a2.set_title(f'Synthetic ({arch_name})')
    sns.heatmap(dc, ax=a3, cmap='Reds', vmin=0, vmax=1, square=True,
                annot=True, fmt='.2f', annot_kws={'size': 7})
    a3.set_title('|Difference|')
    fig.suptitle(f'Correlation — {arch_name}', fontsize=11)
    path = os.path.join(OUT_DIR, f'correlations_{safe_name}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

# TSTR/TRTS bar chart
ARCH_NAMES = list(arch_results.keys())
if ARCH_NAMES:
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x      = np.arange(3)
    width  = 0.22
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
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline\n(real→real)', 'TSTR\n(syn→real)', 'TRTS\n(real→syn)'])
    ax.set_ylim(0, 1.15); ax.set_ylabel('Macro F1')
    ax.axhline(0.5, color='red', linestyle='--', lw=1, label='random')
    ax.legend(); ax.set_title(
        f'TSTR / TRTS / Baseline — ZI Architecture Comparison\n'
        f'(top-{TOP_K_FEATURES} selected features)'
    )
    path = os.path.join(OUT_DIR, 'tstr_trts_comparison.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

# ── Final Summary ──────────────────────────────────
print("\n══════════════════════════════════════════════════")
print(f"  SUMMARY — ZI Architecture Comparison")
print(f"  Feature selection: 16 files × 4 methods, top-{TOP_K_FEATURES} kept")
print(f"  Selected: {selected_features}")
print(f"  Train files ({len(TRAIN_FILES)}): {sorted(TRAIN_FILES)}")
print(f"  Test  files ({len(TEST_FILES)}):  {sorted(TEST_FILES)}")
print("══════════════════════════════════════════════════")

pad = 32
header = f"  {'Metric':<{pad}}"
for a in ARCH_NAMES:
    header += f" {a:>16}"
print(header)
print('-' * (pad + 4 + 16 * len(ARCH_NAMES)))

for cls in [0, 1]:
    label = 'Normal' if cls == 0 else 'Attack'
    ks_mean_row = f"  {'KS mean cls'+str(cls)+' ('+label+')':<{pad}}"
    ks_pass_row = f"  {'KS pass cls'+str(cls)+' ('+label+')':<{pad}}"
    for a in ARCH_NAMES:
        ks_df = arch_results[a]['ks_by_class'][cls]
        ks_mean_row += f" {ks_df['ks_stat'].mean():>16.4f}"
        ks_pass_row += f" {str(ks_df['similar'].sum())+'/'+str(len(ks_df)):>16}"
    print(ks_mean_row)
    print(ks_pass_row)

base_row = f"  {'Baseline F1 (real→real)':<{pad}}"
tstr_row = f"  {'TSTR F1  (syn→real)':<{pad}}"
trts_row = f"  {'TRTS F1  (real→syn)':<{pad}}"
for a in ARCH_NAMES:
    base_row += f" {base_f1:>16.4f}"
    tstr_row += f" {arch_results[a]['tstr_f1']:>16.4f}"
    trts_row += f" {arch_results[a]['trts_f1']:>16.4f}"
print(base_row)
print(tstr_row)
print(trts_row)

# Save metrics JSON
metrics_out = {
    'selected_features'  : selected_features,
    'feature_importance' : df_imp.to_dict(orient='records'),
    'baseline_f1'        : round(base_f1, 4),
    'results'            : {
        a: {
            'tstr_f1': round(arch_results[a]['tstr_f1'], 4),
            'trts_f1': round(arch_results[a]['trts_f1'], 4),
            'ks_mean_cls0': round(arch_results[a]['ks_by_class'][0]['ks_stat'].mean(), 4),
            'ks_mean_cls1': round(arch_results[a]['ks_by_class'][1]['ks_stat'].mean(), 4),
            'ks_pass_cls0': int(arch_results[a]['ks_by_class'][0]['similar'].sum()),
            'ks_pass_cls1': int(arch_results[a]['ks_by_class'][1]['similar'].sum()),
        }
        for a in ARCH_NAMES
    }
}
json_path = os.path.join(OUT_DIR, 'zi_comparison_metrics.json')
with open(json_path, 'w') as f:
    json.dump(metrics_out, f, indent=4)
print(f"\n  Metrics JSON: {json_path}")
print(f"  All results: {OUT_DIR}")
print("══════════════════════════════════════════════════\n")
