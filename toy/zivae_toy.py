# zivae_toy.py
#
# Zero-Inflated VAE (ZI-VAE) — Option 3 fix for sparse/zero-inflated features.
#
# Approach:
#   Each sparse feature gets a TWO-HEAD decoder output:
#     - Bernoulli head  : p(feature == 0)    → BCE loss
#     - Gaussian head   : E[feature | !=0]   → MSE loss (masked to non-zero rows)
#   Dense features use the standard Gaussian head only.
#
#   During synthesis:
#     - Sample Bernoulli gate from the predicted probability
#     - If gate=0 → feature value = 0
#     - If gate=1 → use Gaussian head output
#
# Comparison: ZI-VAE vs plain TimeVAE (Conv1D / Dense path) on same CSV.
# Both are class-conditional (one model per class label).
#
# Outputs:
#   toy/zivae_toy_results/
#     distributions_comparison.png
#     ks_comparison.png
#     correlations_ZI-VAE.png
#     correlations_TimeVAE.png
#     tstr_trts_comparison.png
#
# Run:  python toy/zivae_toy.py

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F_torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)

from time_vae import train_time_vae, synthesize_time_vae   # baseline

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'attack_data',
    'disflooding_var10_base',
    '10_features_timeseries_60_sec.csv'
)
WINDOW_SIZE       = 10
N_SYNTH           = 1000
EPOCHS            = 300
DEVICE            = 'cpu'
SPARSE_THRESH     = 0.30   # fraction of zeros → sparse feature
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'zivae_toy_results')
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════

def load_and_window(csv_path, window_size):
    """
    Returns:
        X_tr, y_tr, X_te, y_te : (N, T, F) float32 normalised [0,1]
        feat_cols               : list of feature column names
        sparse_idx              : list[int] — column indices that are sparse
        g_min, g_max            : pd.Series — used for unnormalising if needed
    """
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    feat_cols = [c for c in df.columns if c != 'label']

    # Detect sparse features (on full data for consistency)
    sparse_cols = [
        c for c in feat_cols
        if (pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]))
        and (df[c] == 0).mean() > SPARSE_THRESH
    ]
    sparse_idx = [feat_cols.index(c) for c in sparse_cols]
    print(f"   Sparse features ({len(sparse_cols)}): {sparse_cols}")

    split  = int(len(df) * 0.8)
    tr, te = df.iloc[:split].copy(), df.iloc[split:].copy()
    g_min  = tr[feat_cols].min()
    g_max  = tr[feat_cols].max()
    denom  = (g_max - g_min).replace(0, 1)

    for d in [tr, te]:
        d[feat_cols] = ((d[feat_cols] - g_min) / denom).clip(0, 1).fillna(0)

    def _windows(d):
        X, y = [], []
        vals   = d[feat_cols].values.astype(np.float32)
        labels = d['label'].values.astype(int)
        for i in range(len(vals) - window_size):
            X.append(vals[i:i + window_size])
            y.append(labels[i + window_size - 1])
        return np.array(X, np.float32), np.array(y, np.int64)

    X_tr, y_tr = _windows(tr)
    X_te, y_te = _windows(te)
    return X_tr, y_tr, X_te, y_te, feat_cols, sparse_idx, g_min, g_max


def ks_table(real_flat, syn_flat, col_names):
    rows = []
    for i, col in enumerate(col_names):
        s, p = stats.ks_2samp(real_flat[:, i], syn_flat[:, i])
        rows.append({'feature': col,
                     'ks_stat': round(float(s), 4),
                     'p_value': round(float(p), 4),
                     'similar': bool(p > 0.05)})
    return pd.DataFrame(rows).sort_values('ks_stat', ascending=False)


def clf_score(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=1000, C=1.0,
                              solver='lbfgs', random_state=42)
    try:
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        return float(accuracy_score(y_te, preds)), \
               float(f1_score(y_te, preds, average='macro', zero_division=0))
    except Exception:
        return 0.0, 0.0


# ═══════════════════════════════════════════════════
# ZI-VAE ARCHITECTURE
# ═══════════════════════════════════════════════════

class ZIVAEEncoder(nn.Module):
    """
    Dense-only encoder (works for any T).
    Input : (B, T, F)  →  flatten  →  MLP  →  (mu, log_var)
    """
    def __init__(self, T: int, F: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.T = T
        self.F = F
        flat = T * F
        self.net = nn.Sequential(
            nn.Linear(flat, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h       = x.flatten(1)          # (B, T*F)
        h       = self.net(h)
        return self.fc_mu(h), self.fc_var(h)


class ZIVAEDecoder(nn.Module):
    """
    Two-head decoder.

    Dense features  → single Gaussian head   (value in [0,1])
    Sparse features → Bernoulli gate head     (is-zero probability, sigmoid)
                    + Gaussian value head     (non-zero magnitude, sigmoid)

    Output (per sample, per timestep):
        - gauss_out  : (B, T, F)       — Gaussian reconstruction for all features
        - gate_logit : (B, T, S)       — logit for P(feature > 0) on sparse dims
          where S = len(sparse_idx)
    """
    def __init__(self, T: int, F: int, latent_dim: int,
                 hidden_dim: int, sparse_idx: list[int]):
        super().__init__()
        self.T          = T
        self.F          = F
        self.sparse_idx = sparse_idx

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, T * F),
        )
        # Bernoulli gate for each sparse feature at each timestep
        if sparse_idx:
            self.gate_net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, T * len(sparse_idx)),
            )
        else:
            self.gate_net = None

    def forward(self, z: torch.Tensor):
        B = z.size(0)
        gauss_out = torch.sigmoid(self.net(z)).view(B, self.T, self.F)

        if self.gate_net is not None:
            gate_logit = self.gate_net(z).view(B, self.T, len(self.sparse_idx))
        else:
            gate_logit = None

        return gauss_out, gate_logit


class ZIVAE(nn.Module):
    def __init__(self, T: int, F: int, latent_dim: int,
                 hidden_dim: int, sparse_idx: list[int]):
        super().__init__()
        self.T          = T
        self.F          = F
        self.latent_dim = latent_dim
        self.sparse_idx = sparse_idx

        self.encoder = ZIVAEEncoder(T, F, latent_dim, hidden_dim)
        self.decoder = ZIVAEDecoder(T, F, latent_dim, hidden_dim, sparse_idx)

    @staticmethod
    def _reparam(mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self._reparam(mu, log_var)
        gauss_out, gate_logit = self.decoder(z)
        return gauss_out, gate_logit, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu') -> np.ndarray:
        """
        Generate n windows.  Returns (n, T, F) float32 in [0,1].

        For each sparse feature at each timestep:
            gate ~ Bernoulli(sigmoid(gate_logit))
            output = gate * gauss_value   (0 if gate=0, value if gate=1)
        """
        z = torch.randn(n, self.latent_dim, device=device)
        gauss_out, gate_logit = self.decoder(z)   # (n,T,F), (n,T,S)

        out = gauss_out.clone()
        if gate_logit is not None:
            gate_prob = torch.sigmoid(gate_logit)               # (n,T,S)
            gate      = torch.bernoulli(gate_prob).bool()       # (n,T,S)
            for s_local, s_global in enumerate(self.sparse_idx):
                out[:, :, s_global] = torch.where(
                    gate[:, :, s_local],
                    gauss_out[:, :, s_global],
                    torch.zeros_like(gauss_out[:, :, s_global])
                )

        return out.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════
# ZI-VAE LOSS
# ═══════════════════════════════════════════════════

def zivae_loss(x          : torch.Tensor,
               gauss_out  : torch.Tensor,
               gate_logit : torch.Tensor | None,
               mu         : torch.Tensor,
               log_var    : torch.Tensor,
               sparse_idx : list[int],
               recon_weight: float = 2.0,
               kl_weight  : float = 1.0,
               free_bits  : float = 0.5,
               ) -> tuple[torch.Tensor, float, float]:
    """
    ELBO with mixed reconstruction loss:

    Dense features  : MSE(gauss_out, x)                per feature
    Sparse features : MSE(gauss_out, x)                 (Gaussian value head)
                    + BCE(sigmoid(gate_logit), gate_label)  (Bernoulli head)
      where gate_label = (x > 0).float()
    """
    T_F  = x.shape[1] * x.shape[2]
    B    = x.shape[0]

    # ── Reconstruction ──────────────────────────────
    # MSE over all features (sum/B then /T_F)
    recon_mse = F_torch.mse_loss(gauss_out, x, reduction='sum') / B

    # Bernoulli gate loss on sparse features
    recon_bce = torch.tensor(0.0, device=x.device)
    if gate_logit is not None and len(sparse_idx) > 0:
        for s_local, s_global in enumerate(sparse_idx):
            # gate label: 1 if non-zero, 0 if zero
            gate_label = (x[:, :, s_global] > 0).float()        # (B, T)
            gl         = gate_logit[:, :, s_local]               # (B, T)
            recon_bce  = recon_bce + F_torch.binary_cross_entropy_with_logits(
                gl, gate_label, reduction='sum'
            ) / B

    recon_total = (recon_weight / T_F) * recon_mse + 0.1 * recon_bce

    # ── KL divergence ───────────────────────────────
    kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl = kl_per_dim.sum(dim=1).mean()

    total = recon_total + kl_weight * kl
    return total, (recon_mse.item() / T_F), kl.item()


# ═══════════════════════════════════════════════════
# ZI-VAE TRAIN
# ═══════════════════════════════════════════════════

def train_zivae(X_np       : np.ndarray,
                T          : int,
                F          : int,
                sparse_idx : list[int],
                latent_dim : int   = 16,
                hidden_dim : int   = 128,
                epochs     : int   = 300,
                batch_size : int   = 128,
                lr         : float = 1e-3,
                recon_weight: float = 2.0,
                kl_warmup  : int   = 100,
                free_bits  : float = 0.5,
                device     : str   = 'cpu') -> ZIVAE:

    X_t    = torch.tensor(X_np, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True,
    )

    model = ZIVAE(T=T, F=F, latent_dim=latent_dim,
                  hidden_dim=hidden_dim, sparse_idx=sparse_idx).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    warmup = max(kl_warmup, 1)

    model.train()
    for epoch in range(epochs):
        kl_weight  = min(1.0, (epoch + 1) / warmup)
        ep_loss = ep_recon = ep_kl = 0.0

        for (batch,) in loader:
            opt.zero_grad()
            gauss_out, gate_logit, mu, log_var = model(batch)
            loss, recon, kl = zivae_loss(
                x           = batch,
                gauss_out   = gauss_out,
                gate_logit  = gate_logit,
                mu          = mu,
                log_var     = log_var,
                sparse_idx  = sparse_idx,
                recon_weight= recon_weight,
                kl_weight   = kl_weight,
                free_bits   = free_bits,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss  += loss.item()
            ep_recon += recon
            ep_kl    += kl

        n_b = max(len(loader), 1)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1:>4}/{epochs}] "
                  f"loss={ep_loss/n_b:.4f}  "
                  f"recon={ep_recon/n_b:.4f}  "
                  f"kl={ep_kl/n_b:.4f}  "
                  f"kl_w={kl_weight:.2f}")

    model.eval()
    return model


# ═══════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════
print("\n── 1. Loading data ──────────────────────────────")
(X_tr, y_tr, X_te, y_te,
 feat_cols, sparse_idx, g_min, g_max) = load_and_window(CSV_PATH, WINDOW_SIZE)

N, T, F = X_tr.shape
print(f"   Train: {X_tr.shape}  Test: {X_te.shape}")
print(f"   Label dist: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

X_tr_flat = X_tr.mean(axis=1)   # (N, F)
X_te_flat = X_te.mean(axis=1)

# shared hyper-params
latent_dim = max(4, min((T * F) // 10, 32))
hidden_dim = max(64, T * F * 4)
free_bits  = round(min(0.5, 8.0 / latent_dim), 4)
batch_size = min(256, max(32, N // 20))
print(f"   latent={latent_dim}  hidden={hidden_dim}  "
      f"free_bits={free_bits}  batch={batch_size}")


# ═══════════════════════════════════════════════════
# 2. TRAIN
# ═══════════════════════════════════════════════════
results = {}

for model_name, use_zi in [('ZI-VAE', True), ('TimeVAE', False)]:
    print(f"\n{'='*55}")
    print(f"  Training {model_name}")
    print(f"{'='*55}")

    syn_X_parts, syn_y_parts = [], []

    for cls in sorted(np.unique(y_tr)):
        X_cls = X_tr[y_tr == cls]
        print(f"\n  Class {cls}: {len(X_cls)} windows")
        if len(X_cls) < batch_size:
            print("  [skip] too few samples")
            continue

        if use_zi:
            model = train_zivae(
                X_np        = X_cls,
                T=T, F=F,
                sparse_idx  = sparse_idx,
                latent_dim  = latent_dim,
                hidden_dim  = hidden_dim,
                epochs      = EPOCHS,
                batch_size  = batch_size,
                lr          = 1e-3,
                recon_weight= 2.0,
                kl_warmup   = 100,
                free_bits   = free_bits,
                device      = DEVICE,
            )
            X_syn = model.sample(N_SYNTH, device=DEVICE)   # (N_SYNTH, T, F)

        else:
            model = train_time_vae(
                X_np        = X_cls,
                T=T, F=F,
                latent_dim  = latent_dim,
                hidden_dim  = hidden_dim,
                enc_filters = (32, 64),
                kernel_size = 3,
                epochs      = EPOCHS,
                batch_size  = batch_size,
                lr          = 1e-3,
                recon_weight= 2.0,
                kl_warmup   = 100,
                free_bits   = free_bits,
                device      = DEVICE,
            )
            X_syn = synthesize_time_vae(model, N_SYNTH, device=DEVICE)

        syn_X_parts.append(X_syn)
        syn_y_parts.append(np.full(N_SYNTH, cls, dtype=np.int64))

    X_syn_all  = np.concatenate(syn_X_parts, axis=0)   # (classes*N_SYNTH, T, F)
    y_syn_all  = np.concatenate(syn_y_parts, axis=0)
    X_syn_flat = X_syn_all.mean(axis=1)                 # (classes*N_SYNTH, F)

    ks_df      = ks_table(X_tr_flat, X_syn_flat, feat_cols)
    _, tstr_f1 = clf_score(X_syn_flat, y_syn_all, X_te_flat,  y_te)
    _, trts_f1 = clf_score(X_tr_flat,  y_tr,      X_syn_flat, y_syn_all)
    _, base_f1 = clf_score(X_tr_flat,  y_tr,      X_te_flat,  y_te)

    results[model_name] = dict(
        ks_df=ks_df, tstr_f1=tstr_f1, trts_f1=trts_f1,
        base_f1=base_f1, X_syn_flat=X_syn_flat, y_syn=y_syn_all,
    )

    print(f"\n  {model_name} KS   mean={ks_df['ks_stat'].mean():.4f}  "
          f"pass={ks_df['similar'].sum()}/{len(ks_df)}")
    print(f"  {model_name} TSTR f1={tstr_f1:.4f}  "
          f"TRTS f1={trts_f1:.4f}  Baseline f1={base_f1:.4f}")


# ═══════════════════════════════════════════════════
# 3. PLOTS
# ═══════════════════════════════════════════════════
print("\n── 3. Saving plots ──────────────────────────────")
model_names = list(results.keys())
colors      = {'ZI-VAE': 'mediumpurple', 'TimeVAE': 'darkorange'}
real_df     = pd.DataFrame(X_tr_flat, columns=feat_cols)

# ── 3a. Distribution overlays ──────────────────────
n_feat = len(feat_cols)
fig, axes = plt.subplots(
    len(model_names), n_feat,
    figsize=(2.8 * n_feat, 3.5 * len(model_names)),
    constrained_layout=True,
)

for row_idx, mname in enumerate(model_names):
    X_syn_flat = results[mname]['X_syn_flat']
    syn_df     = pd.DataFrame(X_syn_flat, columns=feat_cols)
    ks_lu      = results[mname]['ks_df'].set_index('feature')

    for col_idx, col in enumerate(feat_cols):
        ax = axes[row_idx, col_idx]
        r  = real_df[col].values
        s  = syn_df[col].values

        ax.hist(r, bins=25, alpha=0.35, color='steelblue', density=True, label='Real')
        ax.hist(s, bins=25, alpha=0.35, color=colors[mname], density=True, label=mname)
        try:
            xs = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 150)
            ax.plot(xs, stats.gaussian_kde(r)(xs), color='steelblue', lw=1.2)
            ax.plot(xs, stats.gaussian_kde(s)(xs), color=colors[mname], lw=1.2)
        except Exception:
            pass

        ks_v = ks_lu.loc[col, 'ks_stat'] if col in ks_lu.index else float('nan')
        is_sparse = col in [feat_cols[i] for i in sparse_idx]
        title = f"{'*' if is_sparse else ''}{col}\nKS={ks_v:.3f}"
        ax.set_title(title, fontsize=6)
        ax.tick_params(labelsize=5)
        if col_idx == 0:
            ax.set_ylabel(mname, fontsize=7)
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=5)

fig.suptitle('Distribution Overlay — Real vs Synthetic  (* = sparse feature)\n'
             'Row 1: ZI-VAE   Row 2: TimeVAE (baseline)', fontsize=10)
path = os.path.join(OUT_DIR, 'distributions_comparison.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")

# ── 3b. KS bar chart ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)
for ax, mname in zip(axes, model_names):
    ks_df  = results[mname]['ks_df'].sort_values('ks_stat', ascending=False)
    bar_colors = ['tomato' if not r else 'steelblue'
                  for r in ks_df['similar'].tolist()]
    ax.bar(ks_df['feature'], ks_df['ks_stat'], color=bar_colors)
    ax.axhline(0.05, color='black', linestyle='--', lw=1.2)
    ax.set_title(f'{mname}  —  mean KS={ks_df["ks_stat"].mean():.3f}  '
                 f'pass={ks_df["similar"].sum()}/{len(ks_df)}')
    ax.set_ylabel('KS Statistic')
    ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

fig.suptitle('KS Statistic per Feature  (blue=pass p>0.05, red=fail)', fontsize=11)
path = os.path.join(OUT_DIR, 'ks_comparison.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")

# ── 3c. Correlation heatmaps ───────────────────────
for mname in model_names:
    X_syn_flat = results[mname]['X_syn_flat']
    syn_df     = pd.DataFrame(X_syn_flat, columns=feat_cols)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5),
                                      constrained_layout=True)
    rc = real_df.corr()
    sc = syn_df.corr()
    dc = (rc - sc).abs()
    kw = dict(cmap='coolwarm', vmin=-1, vmax=1, square=True,
               annot=True, fmt='.2f', annot_kws={'size': 6},
               xticklabels=True, yticklabels=True)
    sns.heatmap(rc, ax=a1, **kw);  a1.set_title('Real')
    sns.heatmap(sc, ax=a2, **kw);  a2.set_title(f'Synthetic ({mname})')
    sns.heatmap(dc, ax=a3, cmap='Reds', vmin=0, vmax=1, square=True,
                annot=True, fmt='.2f', annot_kws={'size': 6},
                xticklabels=True, yticklabels=True)
    a3.set_title('|Difference|')
    fig.suptitle(f'Correlation — {mname}', fontsize=11)
    safe = mname.replace(' ', '_').replace('-', '')
    path = os.path.join(OUT_DIR, f'correlations_{safe}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path}")

# ── 3d. TSTR / TRTS bar comparison ─────────────────
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
x      = np.arange(3)
width  = 0.35
labels = ['Baseline\n(real→real)', 'TRTS\n(real→syn)', 'TSTR\n(syn→real)']

for i, (mname, color) in enumerate(colors.items()):
    r    = results[mname]
    vals = [r['base_f1'], r['trts_f1'], r['tstr_f1']]
    off  = (i - 0.5) * width
    bars = ax.bar(x + off, vals, width, label=mname, color=color, alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Macro F1')
ax.axhline(0.5, color='red', linestyle='--', lw=1, label='random')
ax.legend()
ax.set_title('TSTR / TRTS / Baseline — ZI-VAE vs TimeVAE')
path = os.path.join(OUT_DIR, 'tstr_trts_comparison.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")


# ═══════════════════════════════════════════════════
# 4. SUMMARY TABLE
# ═══════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════")
print("  FINAL COMPARISON SUMMARY")
print("══════════════════════════════════════════════════")
header = f"{'Metric':<28} {'ZI-VAE':>12} {'TimeVAE':>12}"
print(header)
print('-' * len(header))

metrics = [
    ('Mean KS stat',   lambda r: f"{r['ks_df']['ks_stat'].mean():.4f}"),
    ('KS pass rate',   lambda r: f"{r['ks_df']['similar'].sum()}/{len(r['ks_df'])}"),
    ('Baseline f1',    lambda r: f"{r['base_f1']:.4f}"),
    ('TRTS f1',        lambda r: f"{r['trts_f1']:.4f}"),
    ('TSTR f1',        lambda r: f"{r['tstr_f1']:.4f}"),
    ('TSTR/TRTS gap',  lambda r: f"{abs(r['trts_f1']-r['tstr_f1']):.4f}"),
]

for label, fn in metrics:
    zi_val  = fn(results['ZI-VAE'])
    tv_val  = fn(results['TimeVAE'])
    print(f"  {label:<26} {zi_val:>12} {tv_val:>12}")

print(f"\n  Results saved to: {OUT_DIR}")
print("══════════════════════════════════════════════════\n")
