# run_all_var20.py
#
# Run the feature-selection + ZI generative model comparison experiment
# for all 12 var20 datasets:
#
#   attack_type  ×  suffix
#   ──────────────────────
#   blackhole        base, oo, dec
#   worstparent      base, oo, dec
#   disflooding      base, oo, dec
#   localrepair      base, oo, dec
#
# Each experiment is independent:
#   - Stage 1 : feature selection over 16 files (4 filter/embedded methods)
#   - Stage 2 : ZI-RVAE, ZI-LSTM-Attn, ZI-Transformer trained on top-8 features
#   - Stage 3 : KS, TSTR, TRTS evaluation; plots saved per experiment
#
# All per-experiment results are written to:
#   toy/all_var20_results/<dataset_name>/
#
# A consolidated summary CSV and JSON are written to:
#   toy/all_var20_results/summary.csv
#   toy/all_var20_results/summary.json
#
# Usage:
#   python toy/run_all_var20.py
#
# Estimated runtime: ~20-40 min on CPU (1000 epochs × 2 classes × 3 archs × 12 datasets)
# Set EPOCHS = 300 in the CONFIG section below for a faster run.

import os
import sys
import json
import time
import traceback
import warnings
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

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
from rvae import REncoder

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
ATTACK_TYPES   = ['blackhole', 'worstparent', 'disflooding', 'localrepair']
SUFFIXES       = ['base', 'oo', 'dec']
VAR            = 'var20'

ATTACK_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'attack_data'
)
BASE_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'all_var20_results'
)
os.makedirs(BASE_OUT_DIR, exist_ok=True)

WINDOW_SIZE       = 10
N_SYNTH_PER_CLASS = 1000
EPOCHS            = 1000
DEVICE            = 'cpu'
SPARSE_THRESH     = 0.30
RANK_COLS         = ['rank', 'rank.1']
TOP_K_FEATURES    = 8
FS_N_FILES        = 16   # files used for feature selection
TRAIN_N_FILES     = 14   # files used for model training
# TEST = remaining files (20 - TRAIN_N_FILES)


# ═══════════════════════════════════════════════════
# DATA LOADING + PREPROCESSING
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
    all_feat_cols = [c for c in df_tr.columns if c != 'label']

    rank_cols_present = [c for c in RANK_COLS if c in all_feat_cols]
    for df in [df_tr, df_te]:
        if rank_cols_present:
            diff = df[rank_cols_present].diff().fillna(0)
            df[rank_cols_present] = np.sign(diff) * np.log1p(np.abs(diff))

    sparse_candidates = [
        c for c in all_feat_cols
        if (df_tr[c] == 0).mean() > SPARSE_THRESH
    ]
    for df in [df_tr, df_te]:
        if sparse_candidates:
            df[sparse_candidates] = np.log1p(df[sparse_candidates])

    g_min = df_tr[all_feat_cols].min()
    g_max = df_tr[all_feat_cols].max()
    denom = (g_max - g_min).replace(0, 1)
    for df in [df_tr, df_te]:
        df[all_feat_cols] = ((df[all_feat_cols] - g_min) / denom).clip(0, 1).fillna(0)

    for col in rank_cols_present:
        abs_max = df_tr[col].abs().max()
        if abs_max > 0:
            for df in [df_tr, df_te]:
                df[col] = (df[col] / (2 * abs_max) + 0.5).clip(0, 1)

    use_cols = feat_cols if feat_cols is not None else all_feat_cols
    use_cols = [c for c in use_cols if c in df_tr.columns]
    sparse_idx = [use_cols.index(c) for c in sparse_candidates if c in use_cols]

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

def _normalise(scores):
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.ones_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def feature_importance_one_file(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    feat_cols = [c for c in df.columns if c != 'label']

    rank_cols_present = [c for c in RANK_COLS if c in feat_cols]
    if rank_cols_present:
        diff = df[rank_cols_present].diff().fillna(0)
        df[rank_cols_present] = np.sign(diff) * np.log1p(np.abs(diff))

    sparse_cols = [c for c in feat_cols if (df[c] == 0).mean() > SPARSE_THRESH]
    if sparse_cols:
        df[sparse_cols] = np.log1p(df[sparse_cols])

    g_min = df[feat_cols].min()
    g_max = df[feat_cols].max()
    df[feat_cols] = ((df[feat_cols] - g_min) / (g_max - g_min).replace(0, 1)).clip(0, 1).fillna(0)

    for col in rank_cols_present:
        abs_max = df[col].abs().max()
        if abs_max > 0:
            df[col] = (df[col] / (2 * abs_max) + 0.5).clip(0, 1)

    X = df[feat_cols].values
    y = df['label'].values.astype(int)
    if len(np.unique(y)) < 2:
        return feat_cols, None

    scores = {}
    f_vals, _ = f_classif(X, y)
    scores['anova_f'] = _normalise(np.nan_to_num(f_vals, nan=0.0, posinf=0.0))

    mi = mutual_info_classif(X, y, random_state=0)
    scores['mutual_info'] = _normalise(mi)

    rho = np.array([
        abs(stats.spearmanr(X[:, i], y).statistic)
        if len(np.unique(X[:, i])) > 1 else 0.0
        for i in range(X.shape[1])
    ])
    scores['spearman_rho'] = _normalise(rho)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6,
        class_weight='balanced', random_state=0, n_jobs=-1
    )
    rf.fit(X, y)
    scores['rf_gini'] = _normalise(rf.feature_importances_)

    return feat_cols, scores


def aggregate_feature_importance(fs_files, data_dir):
    all_methods    = ['anova_f', 'mutual_info', 'spearman_rho', 'rf_gini']
    accum          = {m: [] for m in all_methods}
    feat_cols_ref  = None
    n_valid        = 0

    for fname in fs_files:
        path = os.path.join(data_dir, fname)
        feat_cols, sc = feature_importance_one_file(path)
        if feat_cols_ref is None:
            feat_cols_ref = feat_cols
        if sc is None:
            continue
        n_valid += 1
        for m in all_methods:
            accum[m].append(sc[m])

    if n_valid == 0:
        raise RuntimeError('No valid (multi-class) files found for feature selection.')

    avg        = {m: np.mean(accum[m], axis=0) for m in all_methods}
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
    return df_imp, n_valid


# ═══════════════════════════════════════════════════
# STAGE 2 — ZI ARCHITECTURES
# ═══════════════════════════════════════════════════

class ZIDecoder(nn.Module):
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
            gate = torch.bernoulli(torch.sigmoid(gate_logit)).bool()
            for s_local, s_global in enumerate(self.sparse_idx):
                out[:, :, s_global] = torch.where(
                    gate[:, :, s_local],
                    gauss_out[:, :, s_global],
                    torch.zeros_like(gauss_out[:, :, s_global])
                )
        return out


class ZI_RVAE(nn.Module):
    def __init__(self, n_features, window_size, hidden_dim, latent_dim, sparse_idx):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = REncoder(n_features, hidden_dim, latent_dim)
        self.decoder    = ZIDecoder(latent_dim, hidden_dim, n_features,
                                    window_size, sparse_idx)

    def forward(self, x):
        mu, lv = self.encoder(x)
        z      = self.encoder.reparameterize(mu, lv)
        g, gl  = self.decoder(z)
        return g, gl, mu, lv

    @torch.no_grad()
    def sample(self, n, device='cpu'):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder.sample(z).cpu().numpy().astype(np.float32)


class LSTMAttnEncoder_ZI(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim):
        super().__init__()
        self.lstm   = nn.LSTM(n_features, hidden_dim, 1, batch_first=True)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.fc_var.bias, -1.0)

    def forward(self, x):
        out, _ = self.lstm(x)
        w      = torch.softmax(self.attn_w(out), dim=1)
        ctx    = (w * out).sum(dim=1)
        return self.fc_mu(ctx), self.fc_var(ctx)

    @staticmethod
    def reparameterize(mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)


class ZI_LSTMAttn(nn.Module):
    def __init__(self, n_features, window_size, hidden_dim, latent_dim, sparse_idx):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = LSTMAttnEncoder_ZI(n_features, hidden_dim, latent_dim)
        self.decoder    = ZIDecoder(latent_dim, hidden_dim, n_features,
                                    window_size, sparse_idx)

    def forward(self, x):
        mu, lv = self.encoder(x)
        z      = LSTMAttnEncoder_ZI.reparameterize(mu, lv)
        g, gl  = self.decoder(z)
        return g, gl, mu, lv

    @torch.no_grad()
    def sample(self, n, device='cpu'):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder.sample(z).cpu().numpy().astype(np.float32)


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
    def __init__(self, n_features, d_model, latent_dim,
                 n_heads=4, n_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, d_model)
        self.pos_enc     = PositionalEncoding(d_model, dropout=dropout)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc_mu  = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)
        nn.init.constant_(self.fc_var.bias, -1.0)

    def forward(self, x):
        e = self.pos_enc(self.input_proj(x))
        e = self.transformer(e)
        p = e.mean(dim=1)
        return self.fc_mu(p), self.fc_var(p)

    @staticmethod
    def reparameterize(mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)


class ZI_Transformer(nn.Module):
    def __init__(self, n_features, window_size, hidden_dim, latent_dim,
                 sparse_idx, d_model=64, n_heads=4, dim_feedforward=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = TransformerEncoder_ZI(
            n_features, d_model, latent_dim,
            n_heads=n_heads, n_layers=2,
            dim_feedforward=dim_feedforward, dropout=0.1
        )
        self.decoder = ZIDecoder(latent_dim, hidden_dim, n_features,
                                 window_size, sparse_idx)

    def forward(self, x):
        mu, lv = self.encoder(x)
        z      = TransformerEncoder_ZI.reparameterize(mu, lv)
        g, gl  = self.decoder(z)
        return g, gl, mu, lv

    @torch.no_grad()
    def sample(self, n, device='cpu'):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder.sample(z).cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════
# SHARED ZI LOSS + TRAINING LOOP
# ═══════════════════════════════════════════════════

def _cyclical_kl_weight(epoch, n_epochs, n_cycles=4):
    cycle_len = max(n_epochs / n_cycles, 1)
    pos       = (epoch % cycle_len) / cycle_len
    return min(1.0, pos * 2.0)


def zi_loss(x, gauss_out, gate_logit, mu, log_var,
            sparse_idx, kl_weight, loss_factor, free_bits):
    recon_mse = F_torch.mse_loss(gauss_out, x, reduction='mean')
    recon_bce = torch.tensor(0.0, device=x.device)
    if gate_logit is not None and len(sparse_idx) > 0:
        for s_local, s_global in enumerate(sparse_idx):
            recon_bce = recon_bce + F_torch.binary_cross_entropy_with_logits(
                gate_logit[:, :, s_local],
                (x[:, :, s_global] > 0).float(),
                reduction='mean'
            )
    kl_per_dim = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean(dim=0)
    kl_loss    = torch.clamp(kl_per_dim, min=free_bits).sum()
    total      = loss_factor * recon_mse + 0.1 * recon_bce + kl_weight * kl_loss
    return total, recon_mse.item(), kl_loss.item()


def train_zi_model(model, X_np, window_size, n_features, sparse_idx,
                   epochs, batch_size, lr, noise_std, free_bits, n_cycles, device, tag=''):
    X   = X_np.reshape(-1, window_size, n_features).astype(np.float32)
    X_t = torch.tensor(X, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True
    )
    loss_factor = float(window_size * n_features)
    model.train()
    for epoch in range(epochs):
        kl_w = _cyclical_kl_weight(epoch, epochs, n_cycles)
        for (batch,) in loader:
            opt.zero_grad()
            noisy = (batch + noise_std * torch.randn_like(batch)).clamp(0, 1) \
                    if noise_std > 0 else batch
            g, gl, mu, lv = model(noisy)
            loss, _, _ = zi_loss(batch, g, gl, mu, lv,
                                 sparse_idx, kl_w, loss_factor, free_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        if (epoch + 1) % 200 == 0:
            print(f'    [{tag}] epoch {epoch+1}/{epochs}  kl_w={kl_w:.2f}')
    model.eval()
    return model


# ═══════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════

def ks_per_class(real_X, real_y, syn_X, syn_y, feat_cols):
    results = {}
    for cls in [0, 1]:
        r = real_X[real_y == cls].mean(axis=1)
        s = syn_X[syn_y  == cls].mean(axis=1)
        rows = []
        for i, col in enumerate(feat_cols):
            stat, p = stats.ks_2samp(r[:, i], s[:, i])
            rows.append({'feature': col,
                         'ks_stat': round(float(stat), 4),
                         'p_value': round(float(p),    4),
                         'similar': bool(p > 0.05)})
        results[cls] = pd.DataFrame(rows).sort_values('ks_stat', ascending=False)
    return results


class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, 1, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def lstm_score(X_tr, y_tr, X_te, y_te, device='cpu',
               epochs=30, batch_size=128):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long,    device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    clf    = LSTMClassifier(X_tr.shape[2]).to(device)
    opt    = torch.optim.Adam(clf.parameters(), lr=1e-3)
    crit   = nn.CrossEntropyLoss()
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


# ═══════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════

ARCH_COLORS = {
    'ZI-RVAE'       : 'darkorange',
    'ZI-LSTM-Attn'  : 'seagreen',
    'ZI-Transformer': 'mediumpurple',
}


def save_plots(arch_results, feat_cols, X_tr, y_tr, X_te, y_te,
               base_f1, out_dir, dataset_name):

    X_tr_flat = X_tr.mean(axis=1)
    real_df   = pd.DataFrame(X_tr_flat, columns=feat_cols)
    F         = len(feat_cols)
    arch_names = list(arch_results.keys())

    for arch_name, res in arch_results.items():
        X_syn = res['X_syn']; y_syn = res['y_syn']
        ks    = res['ks']; color = ARCH_COLORS[arch_name]
        safe  = arch_name.replace('-', '').replace(' ', '_')

        # KS bar chart
        fig, axes = plt.subplots(1, 2, figsize=(max(10, F), 4),
                                 constrained_layout=True)
        for ax, cls in zip(axes, [0, 1]):
            ks_df = ks[cls].sort_values('ks_stat', ascending=False)
            ax.bar(ks_df['feature'], ks_df['ks_stat'],
                   color=['steelblue' if r else 'tomato' for r in ks_df['similar']])
            ax.axhline(0.05, color='black', linestyle='--', lw=1.2)
            lbl = 'Normal' if cls == 0 else 'Attack'
            ax.set_title(f'cls{cls} ({lbl})  mean_KS={ks_df["ks_stat"].mean():.3f}  '
                         f'pass={ks_df["similar"].sum()}/{len(ks_df)}')
            ax.set_ylim(0, 1.05); ax.set_ylabel('KS Stat')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        fig.suptitle(f'KS — {arch_name} — {dataset_name}', fontsize=10)
        plt.savefig(os.path.join(out_dir, f'ks_{safe}.png'),
                    dpi=100, bbox_inches='tight'); plt.close()

        # Distribution overlays
        fig, axes = plt.subplots(2, F, figsize=(2.5 * F, 6),
                                 constrained_layout=True)
        if F == 1: axes = axes.reshape(2, 1)
        for row, cls in enumerate([0, 1]):
            r_f = X_te[y_te == cls].reshape(-1, F)
            s_f = X_syn[y_syn == cls].reshape(-1, F)
            ks_df = ks[cls].set_index('feature')
            lbl   = 'Normal' if cls == 0 else 'Attack'
            for ci, col in enumerate(feat_cols):
                ax = axes[row, ci]
                r  = r_f[:, ci]; s = s_f[:, ci]
                ax.hist(r, bins=20, alpha=0.35, color='steelblue', density=True)
                ax.hist(s, bins=20, alpha=0.35, color=color,       density=True)
                try:
                    xs = np.linspace(min(r.min(), s.min()),
                                     max(r.max(), s.max()), 120)
                    ax.plot(xs, stats.gaussian_kde(r)(xs), 'steelblue', lw=1)
                    ax.plot(xs, stats.gaussian_kde(s)(xs), color=color, lw=1)
                except Exception:
                    pass
                ks_v = ks_df.loc[col, 'ks_stat'] if col in ks_df.index else float('nan')
                ax.set_title(f'{col}\nKS={ks_v:.3f}', fontsize=6)
                ax.tick_params(labelsize=5)
                if ci == 0: ax.set_ylabel(f'cls{cls} ({lbl})', fontsize=7)
        fig.suptitle(f'Distributions — {arch_name} — {dataset_name}', fontsize=9)
        plt.savefig(os.path.join(out_dir, f'distributions_{safe}.png'),
                    dpi=100, bbox_inches='tight'); plt.close()

        # Correlation heatmap
        syn_df = pd.DataFrame(X_syn.mean(axis=1), columns=feat_cols)
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.5),
                                          constrained_layout=True)
        rc = real_df.corr(); sc = syn_df.corr(); dc = (rc - sc).abs()
        kw = dict(cmap='coolwarm', vmin=-1, vmax=1, square=True,
                  annot=True, fmt='.2f', annot_kws={'size': 6})
        sns.heatmap(rc, ax=a1, **kw); a1.set_title('Real')
        sns.heatmap(sc, ax=a2, **kw); a2.set_title(f'Syn ({arch_name})')
        sns.heatmap(dc, ax=a3, cmap='Reds', vmin=0, vmax=1, square=True,
                    annot=True, fmt='.2f', annot_kws={'size': 6})
        a3.set_title('|Diff|')
        fig.suptitle(f'Correlation — {arch_name} — {dataset_name}', fontsize=9)
        plt.savefig(os.path.join(out_dir, f'correlations_{safe}.png'),
                    dpi=100, bbox_inches='tight'); plt.close()

    # Feature importance bar chart
    imp_path = os.path.join(out_dir, 'feature_importance.png')
    df_imp   = pd.read_csv(os.path.join(out_dir, 'feature_importance.csv'))
    df_plot  = df_imp.sort_values('mean_score', ascending=False)
    methods  = ['anova_f', 'mutual_info', 'spearman_rho', 'rf_gini', 'mean_score']
    titles   = ['ANOVA F', 'Mutual Info', 'Spearman |ρ|', 'RF Gini', 'Mean Score']
    colors   = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple', 'crimson']
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)
    for ax, m, t, c in zip(axes, methods, titles, colors):
        ax.barh(df_plot['feature'], df_plot[m], color=c, alpha=0.8)
        ax.set_title(t, fontsize=9); ax.set_xlim(0, 1.05)
        ax.tick_params(axis='y', labelsize=7); ax.invert_yaxis()
    fig.suptitle(f'Feature Importance — {dataset_name}', fontsize=10)
    plt.savefig(imp_path, dpi=100, bbox_inches='tight'); plt.close()

    # TSTR/TRTS bar chart
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = np.arange(3); width = 0.22
    for i, arch_name in enumerate(arch_names):
        res  = arch_results[arch_name]
        vals = [base_f1, res['tstr_f1'], res['trts_f1']]
        off  = (i - 1) * width
        bars = ax.bar(x + off, vals, width, label=arch_name,
                      color=ARCH_COLORS[arch_name], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline\n(real→real)', 'TSTR\n(syn→real)', 'TRTS\n(real→syn)'])
    ax.set_ylim(0, 1.15); ax.set_ylabel('Macro F1')
    ax.axhline(0.5, color='red', linestyle='--', lw=1, label='random')
    ax.legend(fontsize=8)
    ax.set_title(f'TSTR / TRTS / Baseline — {dataset_name}')
    plt.savefig(os.path.join(out_dir, 'tstr_trts_comparison.png'),
                dpi=100, bbox_inches='tight'); plt.close()


# ═══════════════════════════════════════════════════
# PER-EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════

def run_experiment(dataset_name, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n{"="*64}')
    print(f'  Dataset: {dataset_name}')
    print(f'{"="*64}')

    # ── file split (reproducible, per-dataset seed based on name) ─
    all_files = [f'{i}_features_timeseries_60_sec.csv' for i in range(1, 21)]
    rng       = np.random.default_rng(seed=42)
    shuffled  = rng.permutation(all_files).tolist()
    fs_files  = shuffled[:FS_N_FILES]
    tr_files  = shuffled[:TRAIN_N_FILES]
    te_files  = shuffled[TRAIN_N_FILES:]

    # ── Stage 1: feature selection ───────────────────
    print(f'\n  [1/3] Feature selection over {FS_N_FILES} files...')
    t0    = time.time()
    df_imp, n_valid = aggregate_feature_importance(fs_files, data_dir)
    print(f'        done in {time.time()-t0:.1f}s  ({n_valid} valid files)')
    df_imp.to_csv(os.path.join(out_dir, 'feature_importance.csv'), index=False)

    selected = df_imp.head(TOP_K_FEATURES)['feature'].tolist()
    print(f'  Selected top-{TOP_K_FEATURES}: {selected}')

    # ── Load + window with selected features ─────────
    df_tr = load_files(tr_files, data_dir)
    df_te = load_files(te_files, data_dir)
    X_tr, y_tr, X_te, y_te, feat_cols, sparse_idx = preprocess_and_window(
        df_tr, df_te, WINDOW_SIZE, feat_cols=selected
    )
    N, T, F = X_tr.shape
    print(f'  Windows: train={X_tr.shape}  test={X_te.shape}')
    print(f'  Sparse idx: {sparse_idx} → {[feat_cols[i] for i in sparse_idx]}')
    print(f'  Class dist train: {dict(zip(*np.unique(y_tr, return_counts=True)))}')
    print(f'  Class dist test : {dict(zip(*np.unique(y_te, return_counts=True)))}')

    # Hyperparams
    latent_dim = max(4, min((T * F) // 10, 32))
    hidden_dim = max(64, F * 8)
    free_bits  = round(min(0.5, 8.0 / latent_dim), 4)
    batch_size = min(256, max(32, N // 20))
    d_model    = max(32, (latent_dim // 4) * 4)

    # ── Baseline ─────────────────────────────────────
    print(f'\n  [2/3] Training models...')
    base_f1 = lstm_score(X_tr, y_tr, X_te, y_te, device=DEVICE)
    print(f'  Baseline F1 (real→real): {base_f1:.4f}')

    ARCH_DEFS = {
        'ZI-RVAE': lambda: ZI_RVAE(F, T, hidden_dim, latent_dim, sparse_idx),
        'ZI-LSTM-Attn': lambda: ZI_LSTMAttn(F, T, hidden_dim, latent_dim, sparse_idx),
        'ZI-Transformer': lambda: ZI_Transformer(
            F, T, hidden_dim, latent_dim, sparse_idx,
            d_model=d_model, n_heads=4, dim_feedforward=hidden_dim
        ),
    }

    arch_results = {}
    for arch_name, model_fn in ARCH_DEFS.items():
        print(f'\n  ── {arch_name} ──')
        syn_X_parts, syn_y_parts = [], []

        for cls in [0, 1]:
            X_cls = X_tr[y_tr == cls]
            label = 'Normal' if cls == 0 else 'Attack'
            print(f'    cls{cls} ({label}): {len(X_cls)} windows', end='  ')
            if len(X_cls) < batch_size:
                print(f'[skip: too few]')
                continue

            model = model_fn().to(DEVICE)
            t0    = time.time()
            model = train_zi_model(
                model=model, X_np=X_cls.reshape(len(X_cls), -1),
                window_size=T, n_features=F, sparse_idx=sparse_idx,
                epochs=EPOCHS, batch_size=batch_size, lr=1e-3,
                noise_std=0.05, free_bits=free_bits, n_cycles=4,
                device=DEVICE, tag=f'{arch_name} cls{cls}'
            )
            print(f'trained in {time.time()-t0:.0f}s')

            X_syn = model.sample(N_SYNTH_PER_CLASS, device=DEVICE)
            syn_X_parts.append(X_syn)
            syn_y_parts.append(np.full(N_SYNTH_PER_CLASS, cls, dtype=np.int64))

        if not syn_X_parts:
            print(f'  [skip {arch_name}: no valid classes]')
            continue

        X_syn_all = np.concatenate(syn_X_parts)
        y_syn_all = np.concatenate(syn_y_parts)

        ks   = ks_per_class(X_tr, y_tr, X_syn_all, y_syn_all, feat_cols)
        tstr = lstm_score(X_syn_all, y_syn_all, X_te,      y_te,      device=DEVICE)
        trts = lstm_score(X_tr,      y_tr,      X_syn_all, y_syn_all, device=DEVICE)

        arch_results[arch_name] = {
            'X_syn': X_syn_all, 'y_syn': y_syn_all,
            'ks': ks, 'tstr_f1': tstr, 'trts_f1': trts,
        }

        for cls in [0, 1]:
            ks_df = ks[cls]; lbl = 'Normal' if cls == 0 else 'Attack'
            print(f'  [{arch_name}] cls{cls} ({lbl}): '
                  f'KS mean={ks_df["ks_stat"].mean():.4f}  '
                  f'pass={ks_df["similar"].sum()}/{len(ks_df)}')
        print(f'  [{arch_name}] TSTR={tstr:.4f}  TRTS={trts:.4f}  '
              f'Baseline={base_f1:.4f}')

    # ── Stage 3: plots ────────────────────────────────
    print(f'\n  [3/3] Saving plots...')
    save_plots(arch_results, feat_cols, X_tr, y_tr, X_te, y_te,
               base_f1, out_dir, dataset_name)

    # ── Save per-experiment JSON ──────────────────────
    arch_names = list(arch_results.keys())
    metrics = {
        'dataset'           : dataset_name,
        'selected_features' : selected,
        'feature_importance': df_imp.to_dict(orient='records'),
        'n_train_windows'   : int(N),
        'n_test_windows'    : int(len(y_te)),
        'baseline_f1'       : round(base_f1, 4),
        'results'           : {
            a: {
                'tstr_f1'     : round(arch_results[a]['tstr_f1'], 4),
                'trts_f1'     : round(arch_results[a]['trts_f1'], 4),
                'ks_mean_cls0': round(arch_results[a]['ks'][0]['ks_stat'].mean(), 4),
                'ks_mean_cls1': round(arch_results[a]['ks'][1]['ks_stat'].mean(), 4),
                'ks_pass_cls0': int(arch_results[a]['ks'][0]['similar'].sum()),
                'ks_pass_cls1': int(arch_results[a]['ks'][1]['similar'].sum()),
            }
            for a in arch_names
        }
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'  Saved to: {out_dir}')
    return metrics


# ═══════════════════════════════════════════════════
# MAIN — iterate all 12 datasets
# ═══════════════════════════════════════════════════

DATASETS = [
    f'{attack}_{VAR}_{suffix}'
    for attack in ATTACK_TYPES
    for suffix in SUFFIXES
]

print(f'\n{"="*64}')
print(f'  Running {len(DATASETS)} experiments')
print(f'  Datasets: {DATASETS}')
print(f'  EPOCHS={EPOCHS}  TOP_K={TOP_K_FEATURES}  WINDOW={WINDOW_SIZE}')
print(f'{"="*64}')

all_metrics = []
failed      = []

for dataset_name in DATASETS:
    data_dir = os.path.join(ATTACK_DATA_ROOT, dataset_name)
    out_dir  = os.path.join(BASE_OUT_DIR, dataset_name)

    if not os.path.isdir(data_dir):
        print(f'\n  [SKIP] {dataset_name} — directory not found')
        failed.append({'dataset': dataset_name, 'error': 'directory not found'})
        continue

    try:
        metrics = run_experiment(dataset_name, data_dir, out_dir)
        all_metrics.append(metrics)
    except Exception as e:
        print(f'\n  [ERROR] {dataset_name}: {e}')
        traceback.print_exc()
        failed.append({'dataset': dataset_name, 'error': str(e)})

# ═══════════════════════════════════════════════════
# CONSOLIDATED SUMMARY
# ═══════════════════════════════════════════════════

print(f'\n{"="*64}')
print(f'  Building consolidated summary ({len(all_metrics)} experiments)')
print(f'{"="*64}')

summary_rows = []
for m in all_metrics:
    row = {
        'dataset'    : m['dataset'],
        'n_train'    : m['n_train_windows'],
        'n_test'     : m['n_test_windows'],
        'baseline_f1': m['baseline_f1'],
        'selected_features': ', '.join(m['selected_features']),
    }
    for arch, res in m['results'].items():
        prefix = arch.replace(' ', '_').replace('-', '')
        row[f'{prefix}_tstr'] = res['tstr_f1']
        row[f'{prefix}_trts'] = res['trts_f1']
        row[f'{prefix}_ks_mean_cls0'] = res['ks_mean_cls0']
        row[f'{prefix}_ks_mean_cls1'] = res['ks_mean_cls1']
        row[f'{prefix}_ks_pass_cls0'] = res['ks_pass_cls0']
        row[f'{prefix}_ks_pass_cls1'] = res['ks_pass_cls1']
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
csv_path   = os.path.join(BASE_OUT_DIR, 'summary.csv')
json_path  = os.path.join(BASE_OUT_DIR, 'summary.json')
df_summary.to_csv(csv_path, index=False)
with open(json_path, 'w') as f:
    json.dump({'experiments': all_metrics, 'failed': failed}, f, indent=4)

print(f'\n  Summary CSV : {csv_path}')
print(f'  Summary JSON: {json_path}')
if failed:
    print(f'\n  Failed datasets ({len(failed)}):')
    for e in failed:
        print(f'    {e["dataset"]}: {e["error"]}')

# ── Cross-dataset comparison plot ──────────────────
if len(df_summary) >= 2:
    arch_cols = {
        'ZI-RVAE'       : 'ZIRVAE',
        'ZI-LSTM-Attn'  : 'ZILSTMAttn',
        'ZI-Transformer': 'ZITransformer',
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(df_summary) * 0.5 + 1)),
                              constrained_layout=True)
    for ax, metric in zip(axes, ['tstr', 'trts']):
        title = 'TSTR (syn→real)' if metric == 'tstr' else 'TRTS (real→syn)'
        y_pos = np.arange(len(df_summary))
        width = 0.22
        for i, (arch_label, arch_prefix) in enumerate(arch_cols.items()):
            col  = f'{arch_prefix}_{metric}'
            vals = df_summary[col].values if col in df_summary.columns else np.zeros(len(df_summary))
            off  = (i - 1) * width
            ax.barh(y_pos + off, vals, width,
                    label=arch_label, color=ARCH_COLORS[arch_label], alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_summary['dataset'], fontsize=7)
        ax.set_xlabel('Macro F1'); ax.set_xlim(0, 1.05)
        ax.axvline(0.5, color='red', linestyle='--', lw=1)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        # baseline markers
        for j, bline in enumerate(df_summary['baseline_f1'].values):
            ax.plot(bline, j, 'k|', markersize=8, markeredgewidth=1.5)
    fig.suptitle(f'ZI Architecture Comparison — All {VAR} Datasets\n'
                 f'(black tick = Baseline F1)', fontsize=11)
    cmp_path = os.path.join(BASE_OUT_DIR, 'cross_dataset_comparison.png')
    plt.savefig(cmp_path, dpi=120, bbox_inches='tight'); plt.close()
    print(f'  Cross-dataset plot: {cmp_path}')

# ── Print final table ───────────────────────────────
print(f'\n{"="*64}')
print(f'  FINAL SUMMARY — {len(all_metrics)} experiments')
print(f'{"="*64}')
if not df_summary.empty:
    cols_show = ['dataset', 'baseline_f1',
                 'ZIRVAE_tstr', 'ZIRVAE_trts',
                 'ZILSTMAttn_tstr', 'ZILSTMAttn_trts',
                 'ZITransformer_tstr', 'ZITransformer_trts']
    cols_show = [c for c in cols_show if c in df_summary.columns]
    print(df_summary[cols_show].to_string(index=False))
print(f'\nAll results in: {BASE_OUT_DIR}')
print('Done.\n')
