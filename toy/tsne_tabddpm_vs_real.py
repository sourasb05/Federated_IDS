"""
t-SNE comparison: TabDDPM generated attack samples vs real attack samples
Domain: localrepair_var20_dec  |  Client 0

Overlap metrics printed:
  - Precision  (fidelity):  % synthetic inside real manifold
  - Recall     (coverage):  % real covered by synthetic manifold
  - Alpha-Precision / Alpha-Recall (k-NN based, k=5)

Usage:
    python toy/tsne_tabddpm_vs_real.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR       = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR      = os.path.join(PROJECT_ROOT, 'attack_data', 'localrepair_var15_base')
CKPT_PATH     = os.path.join(PROJECT_ROOT, 'saved_models_efl', 'localrepair',
                              'tabddpm_client4_localrepair_var15_base_attack.pt')
OUT_PATH      = os.path.join(PROJECT_ROOT, 'toy', 'tsne_tabddpm_localrepair_var15_base.png')

sys.path.insert(0, SRC_DIR)

# ── colour-blind friendly palette (Wong 2011) ────────────────────────────────
REAL_COLOR  = '#0072B2'   # blue
SYN_COLOR   = '#E69F00'   # orange
REAL_MARKER = 'o'
SYN_MARKER  = 's'

# ── config ───────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 10
N_FEATURES   = 14
N_GENERATE   = 1000       # synthetic samples to generate
N_REAL_MAX   = 1000       # cap real samples for balance
TSNE_PERP    = 30
TSNE_ITER    = 1000
RANDOM_STATE = 42
KNN_K        = 5         # neighbours for precision/recall


# ── overlap metrics ──────────────────────────────────────────────────────────
MMD_CAP = 2000   # max rows for MMD² (O(N²) — cap to stay fast)

def compute_precision_recall(X_real, X_syn, k=5):
    """
    Improved Precision & Recall (Kynkäänniemi et al. 2019).
    Uses ball_tree for speed on low-dim data.
    """
    nn_real = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(X_real)
    dists_rr, _ = nn_real.kneighbors(X_real)
    radii_real   = dists_rr[:, -1]

    nn_syn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(X_syn)
    dists_ss, _ = nn_syn.kneighbors(X_syn)
    radii_syn    = dists_ss[:, -1]

    # Precision: each synthetic point inside any real ball?
    dists_sr, _  = nn_real.kneighbors(X_syn)
    nearest_real = dists_sr[:, 0]
    _, nn_idx    = nn_real.kneighbors(X_syn, n_neighbors=1)
    in_real_ball = nearest_real <= radii_real[nn_idx[:, 0]]
    precision    = in_real_ball.mean()

    # Recall: each real point inside any synthetic ball?
    dists_rs, _  = nn_syn.kneighbors(X_real)
    nearest_syn  = dists_rs[:, 0]
    _, nn_idx_s  = nn_syn.kneighbors(X_real, n_neighbors=1)
    in_syn_ball  = nearest_syn <= radii_syn[nn_idx_s[:, 0]]
    recall       = in_syn_ball.mean()

    return float(precision), float(recall)


def compute_mmd_squared(X_real, X_syn, gamma=1.0, cap=MMD_CAP, seed=42):
    """RBF-kernel MMD² — subsampled to `cap` rows to keep O(N²) fast."""
    rng = np.random.default_rng(seed)
    if len(X_real) > cap:
        X_real = X_real[rng.choice(len(X_real), cap, replace=False)]
    if len(X_syn) > cap:
        X_syn  = X_syn[rng.choice(len(X_syn),  cap, replace=False)]

    def rbf(A, B):
        sq = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * sq)

    Krr  = rbf(X_real, X_real)
    Kss  = rbf(X_syn,  X_syn)
    Krs  = rbf(X_real, X_syn)
    return float(Krr.mean() + Kss.mean() - 2 * Krs.mean())


# ── 1. Load real attack data ─────────────────────────────────────────────────
print("Loading real attack data …")

def load_and_window(data_dir, window_size, n_features):
    DROP  = ['Unnamed: 0']
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))
    dfs   = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        for c in DROP:
            if c in df.columns:
                df = df.drop(columns=[c])
        dfs.append(df)

    df_all    = pd.concat(dfs, ignore_index=True)
    feat_cols = [c for c in df_all.columns if c != 'label']

    gmin  = df_all[feat_cols].min()
    gmax  = df_all[feat_cols].max()
    denom = (gmax - gmin).replace(0, 1)
    df_all[feat_cols] = (df_all[feat_cols] - gmin) / denom
    df_all[feat_cols] = df_all[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    attack_df = df_all[df_all['label'] == 1].reset_index(drop=True)
    X = attack_df[feat_cols].values.astype(np.float32)

    windows = []
    for i in range(0, len(X) - window_size + 1, 1):
        windows.append(X[i:i + window_size])
    return np.array(windows, dtype=np.float32)   # (N, T, F)


X_real = load_and_window(DATA_DIR, WINDOW_SIZE, N_FEATURES)
print(f"  Real attack windows: {X_real.shape}")

rng = np.random.default_rng(RANDOM_STATE)
if len(X_real) > N_REAL_MAX:
    idx    = rng.choice(len(X_real), N_REAL_MAX, replace=False)
    X_real = X_real[idx]
print(f"  After subsampling  : {X_real.shape}")


# ── 2. Load TabDDPM and generate synthetic attack samples ────────────────────
print("\nLoading TabDDPM checkpoint …")
from models import TabDDPMGenerator

ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
gen  = TabDDPMGenerator(
    data_dim    = ckpt['data_dim'],
    window_size = ckpt['window_size'],
    n_features  = ckpt['n_features'],
    n_steps     = ckpt['n_steps'],
).to('cpu')
gen.denoiser.load_state_dict(ckpt['denoiser_state_dict'])
gen.eval()

print(f"  Generating {N_GENERATE} synthetic attack samples …")
X_syn = gen.generate(n=N_GENERATE, device='cpu')   # (N, T, F)
print(f"  Synthetic shape: {X_syn.shape}")


# ── 3. Flatten (N, T, F) → (N, T*F) ─────────────────────────────────────────
X_real_flat = X_real.reshape(len(X_real), -1)
X_syn_flat  = X_syn.reshape(len(X_syn),   -1)


# ── 4. Quantitative overlap metrics ──────────────────────────────────────────
# Metrics computed on per-timestep rows (N*T, F) instead of flattened (N, T*F)
# because k-NN in 140-dim is too sparse to be meaningful.
print("\nComputing overlap metrics (per-timestep feature space, N×14) …")

X_real_ts = X_real.reshape(-1, N_FEATURES)   # (N*T, F)
X_syn_ts  = X_syn.reshape(-1,  N_FEATURES)

precision, recall = compute_precision_recall(X_real_ts, X_syn_ts, k=KNN_K)
f1_pr  = 2 * precision * recall / (precision + recall + 1e-9)
mmd2   = compute_mmd_squared(X_real_ts, X_syn_ts)

print("\n" + "=" * 58)
print(f"  Overlap Metrics  (per-timestep space 14-dim, k-NN k={KNN_K})")
print("=" * 58)
print(f"  Precision  (fidelity  — syn inside real) : {precision*100:6.1f}%")
print(f"  Recall     (coverage  — real covered)    : {recall*100:6.1f}%")
print(f"  F1 score   (harmonic mean)               : {f1_pr*100:6.1f}%")
print(f"  MMD²       (distribution distance)       : {mmd2:.6f}")
print("=" * 58)
print()
print("Interpretation:")
print(f"  • {precision*100:.1f}% of synthetic timesteps are statistically 'plausible'")
print(f"    (fall within the real data neighbourhood).")
print(f"  • {recall*100:.1f}% of real attack timesteps are reproduced by the generator.")
print(f"  • Combined F1 overlap: {f1_pr*100:.1f}%")
print()
print("Note: t-SNE plot uses flattened windows (N×140) for sequence-level view.")


# ── 5. t-SNE ─────────────────────────────────────────────────────────────────
X_all  = np.concatenate([X_real_flat, X_syn_flat], axis=0)
labels = np.array(['Real'] * len(X_real_flat) + ['Synthetic'] * len(X_syn_flat))

print(f"\nRunning t-SNE on {len(X_all)} samples "
      f"({len(X_real_flat)} real + {len(X_syn_flat)} synthetic) …")

tsne = TSNE(n_components=2, perplexity=TSNE_PERP, max_iter=TSNE_ITER,
            random_state=RANDOM_STATE, n_jobs=-1)
Z    = tsne.fit_transform(X_all)

Z_real = Z[labels == 'Real']
Z_syn  = Z[labels == 'Synthetic']


# ── 6. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(Z_real[:, 0], Z_real[:, 1],
           c=REAL_COLOR, marker=REAL_MARKER,
           s=25, alpha=0.6, linewidths=0.3,
           edgecolors='white')

ax.scatter(Z_syn[:, 0], Z_syn[:, 1],
           c=SYN_COLOR, marker=SYN_MARKER,
           s=25, alpha=0.6, linewidths=0.3,
           edgecolors='white')

# stats annotation in the plot
stats_text = (
    f"Precision (fidelity):  {precision*100:.1f}%\n"
    f"Recall    (coverage):  {recall*100:.1f}%\n"
    f"F1 overlap:            {f1_pr*100:.1f}%\n"
    f"MMD²:                  {mmd2:.5f}"
)
ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

patches = [
    mpatches.Patch(color=REAL_COLOR, label=f'Real attack  (n={len(Z_real)})'),
    mpatches.Patch(color=SYN_COLOR,  label=f'TabDDPM synthetic  (n={len(Z_syn)})'),
]
ax.legend(handles=patches, fontsize=11, framealpha=0.9, loc='lower right')

ax.set_title('t-SNE: Real vs TabDDPM Synthetic Attack Samples\n'
             'localrepair_var20_dec  |  Client 0', fontsize=13)
ax.set_xlabel('t-SNE dim 1', fontsize=11)
ax.set_ylabel('t-SNE dim 2', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved → {OUT_PATH}")
plt.show()
