"""
rank_distribution_check.py
───────────────────────────
Compare real vs synthetic rank / rank.1 distributions for blackhole_var10_base.

Produces two figures saved to blackhole_perturb_results_new approach/blackhole_var10_base/:
  1. rank_kde_hist.png   — 2×2 histogram + KDE overlay (one panel per col × class)
  2. rank_qqplot.png     — 2×2 Q-Q plot (quantile-quantile, real vs synthetic)

Run:
    conda run -n vinnova python toy/rank_distribution_check.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
from scipy.stats import ks_2samp

warnings.filterwarnings('ignore')

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(SCRIPT_DIR, '..', 'attack_data', 'blackhole_var10_base')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'blackhole_perturb_results_new approach',
                           'blackhole_var10_base')
OUT_DIR     = RESULTS_DIR
os.makedirs(OUT_DIR, exist_ok=True)

RANK_COLS     = ['rank', 'rank.1']
DEAD_FEATURES = ['disr', 'diss', 'disr.1', 'diss.1']
WINDOW_SIZE   = 10
N_SYNTH       = 1000
RNG_SEED      = 42
SPARSE_THRESH = 0.30

# ── reproduce the exact train/test split used in blackhole_perturb.py ────────
all_files = sorted([f for f in os.listdir(DATA_ROOT) if f.endswith('.csv')])
rng_split = np.random.default_rng(seed=RNG_SEED)
shuffled  = rng_split.permutation(all_files).tolist()

def _load(files):
    dfs = []
    for fname in files:
        df = pd.read_csv(os.path.join(DATA_ROOT, fname),
                         encoding='utf-8', encoding_errors='ignore')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df_tr_raw = _load(shuffled[:14])
df_te_raw = _load(shuffled[14:])

# ── preprocess exactly as in blackhole_perturb.py (no differencing on rank) ──
def preprocess(df_tr, df_te):
    feat_cols    = [c for c in df_tr.columns if c != 'label']
    rank_present = [c for c in RANK_COLS if c in feat_cols]
    sparse_cols  = [c for c in feat_cols
                    if c not in rank_present and (df_tr[c] == 0).mean() > SPARSE_THRESH]
    for df in [df_tr, df_te]:
        if sparse_cols:
            df[sparse_cols] = np.log1p(df[sparse_cols])
    g_min  = df_tr[feat_cols].min()
    g_max  = df_tr[feat_cols].max()
    denom  = (g_max - g_min).replace(0, 1)
    for df in [df_tr, df_te]:
        df[feat_cols] = ((df[feat_cols] - g_min) / denom).clip(0, 1).fillna(0)
    return df_tr, df_te, dict(g_min=g_min, g_max=g_max, denom=denom)

df_tr, df_te, preproc = preprocess(df_tr_raw.copy(), df_te_raw.copy())

# ── window the test set (same logic as blackhole_perturb.py) ─────────────────
def make_windows(df):
    feat_cols = [c for c in df.columns if c != 'label']
    vals   = df[feat_cols].values.astype(np.float32)
    labels = df['label'].values.astype(int)
    X, y   = [], []
    for i in range(len(vals) - WINDOW_SIZE):
        lbl = 1 if 1 in labels[i:i + WINDOW_SIZE] else 0
        X.append(vals[i:i + WINDOW_SIZE])
        y.append(lbl)
    return np.array(X, np.float32), np.array(y, np.int64), feat_cols

X_te, y_te, feat_cols = make_windows(df_te)
rank_idx  = feat_cols.index('rank')
rank1_idx = feat_cols.index('rank.1')

# real test windows per class — flatten over timesteps → (N*T,)
real = {}
for cls, name in [(0, 'normal'), (1, 'attack')]:
    mask = y_te == cls
    real[name] = {
        'rank':   X_te[mask, :, rank_idx].ravel(),
        'rank.1': X_te[mask, :, rank1_idx].ravel(),
    }

# ── AR(1) fitting (raw space, training data) ──────────────────────────────────
def fit_ar1(seqs):
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

ar1_params = {}
for col in RANK_COLS:
    ar1_params[col] = {}
    for cls, name in [(0, 'normal'), (1, 'attack')]:
        vals = df_tr_raw.loc[df_tr_raw['label'] == cls, col].values.astype(np.float64)
        n_win = len(vals) // WINDOW_SIZE
        seqs  = vals[:n_win * WINDOW_SIZE].reshape(n_win, WINDOW_SIZE)
        ar1_params[col][name] = fit_ar1(seqs)
        p = ar1_params[col][name]
        print(f"  AR(1) {col} {name}: mu={p['mu']:.1f}  std={p['std']:.1f}  phi={p['phi']:.4f}  sigma_inn={p['sigma_inn']:.2f}")

# ── sample AR(1) sequences ────────────────────────────────────────────────────
def sample_ar1(params, n_windows, rng):
    mu, phi = params['mu'], params['phi']
    sigma_inn, std = params['sigma_inn'], params['std']
    out = np.empty((n_windows, WINDOW_SIZE), dtype=np.float32)
    out[:, 0] = rng.normal(mu, std, size=n_windows).astype(np.float32)
    for t in range(1, WINDOW_SIZE):
        noise   = rng.normal(0, sigma_inn, size=n_windows).astype(np.float32)
        out[:, t] = phi * (out[:, t-1] - mu) + mu + noise
    return out   # raw rank space

def raw_to_norm(raw_seqs, col):
    g_min = float(preproc['g_min'][col])
    denom = float(preproc['denom'][col])
    # round to integer grid before normalising — rank is a discrete hop count
    return np.clip((np.round(raw_seqs).astype(np.float64) - g_min) / denom, 0, 1).astype(np.float32)

rng = np.random.default_rng(RNG_SEED)
synth = {}
for name in ['normal', 'attack']:
    synth[name] = {}
    for col in RANK_COLS:
        raw_seqs = sample_ar1(ar1_params[col][name], N_SYNTH, rng)
        synth[name][col] = raw_to_norm(raw_seqs, col).ravel()

# ── pretty names ──────────────────────────────────────────────────────────────
COL_LABEL = {'rank': 'rank', 'rank.1': 'rank.1'}
CLASS_COLOR = {
    'real_normal':  '#2166ac',   # blue
    'real_attack':  '#d73027',   # red
    'synth_normal': '#74add1',   # light blue
    'synth_attack': '#f46d43',   # light orange
}

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — 2×2  Histogram (real) + KDE (synthetic)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('rank & rank.1 — Real (hist) vs Synthetic AR(1) (KDE)\nblackhole_var10_base',
             fontsize=13, fontweight='bold', y=1.01)

panel_cfg = [
    (0, 0, 'rank',   'normal'),
    (0, 1, 'rank',   'attack'),
    (1, 0, 'rank.1', 'normal'),
    (1, 1, 'rank.1', 'attack'),
]

for row, col_idx, feat, cls_name in panel_cfg:
    ax = axes[row][col_idx]

    r = real[cls_name][feat]
    s = synth[cls_name][feat]

    real_color  = CLASS_COLOR[f'real_{cls_name}']
    synth_color = CLASS_COLOR[f'synth_{cls_name}']

    # shared bin edges so real and synthetic use identical bins
    n_bins   = min(80, max(30, int(np.sqrt(len(r)))))
    lo       = min(r.min(), s.min()) - 0.01
    hi       = max(r.max(), s.max()) + 0.01
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    # histogram of real values
    ax.hist(r, bins=bin_edges, density=True, alpha=0.50, color=real_color,
            label=f'Real {cls_name} (n={len(r):,})', edgecolor='none')

    # histogram of synthetic values (same bins) — shows discrete spikes
    ax.hist(s, bins=bin_edges, density=True, alpha=0.45, color=synth_color,
            label=f'Synth {cls_name} AR(1) (n={len(s):,})', edgecolor='none')

    # mean lines
    ax.axvline(r.mean(), color=real_color,  lw=1.4, ls='--', alpha=0.85,
               label=f'Real μ = {r.mean():.3f}')
    ax.axvline(s.mean(), color=synth_color, lw=1.4, ls=':',  alpha=0.85,
               label=f'Synth μ = {s.mean():.3f}')

    # KS statistic
    ks_stat, ks_p = ks_2samp(r, s)
    ax.set_title(f'{feat}  —  {cls_name}\nKS = {ks_stat:.3f}  (p = {ks_p:.2e})',
                 fontsize=10)
    ax.set_xlabel('Normalised value [0, 1]', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.tick_params(labelsize=8)
    sns_style = dict(color='0.85', linewidth=0.5)
    ax.grid(True, **sns_style)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'rank_kde_hist.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — 2×2  Q-Q plot  (real quantiles vs synthetic quantiles)
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
fig2.suptitle('rank & rank.1 — Q-Q: Real quantiles vs Synthetic AR(1) quantiles\nblackhole_var10_base',
              fontsize=13, fontweight='bold', y=1.01)

for row, col_idx, feat, cls_name in panel_cfg:
    ax = axes2[row][col_idx]

    r = real[cls_name][feat]
    s = synth[cls_name][feat]

    # compute quantiles at same probability points
    probs    = np.linspace(0, 100, 200)
    q_real   = np.percentile(r, probs)
    q_synth  = np.percentile(s, probs)

    dot_color = CLASS_COLOR[f'real_{cls_name}']

    ax.scatter(q_real, q_synth, s=12, alpha=0.7, color=dot_color, linewidths=0)

    # 45° reference line
    lo = min(q_real.min(), q_synth.min())
    hi = max(q_real.max(), q_synth.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y = x (perfect match)')

    ks_stat, _ = ks_2samp(r, s)
    ax.set_title(f'{feat}  —  {cls_name}  |  KS = {ks_stat:.3f}', fontsize=10)
    ax.set_xlabel('Real quantiles', fontsize=9)
    ax.set_ylabel('Synthetic quantiles', fontsize=9)
    ax.legend(fontsize=7.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, color='0.85', linewidth=0.5)
    ax.set_aspect('equal', adjustable='datalim')

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'rank_qqplot.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {out2}")

# ── print summary table ───────────────────────────────────────────────────────
print("\n── KS summary ──────────────────────────────────────")
print(f"{'feature':<10} {'class':<8} {'real μ':>8} {'synth μ':>8} {'real σ':>8} {'synth σ':>8} {'KS':>7}")
print("-" * 60)
for feat in RANK_COLS:
    for cls_name in ['normal', 'attack']:
        r = real[cls_name][feat]
        s = synth[cls_name][feat]
        ks, _ = ks_2samp(r, s)
        print(f"{feat:<10} {cls_name:<8} {r.mean():>8.4f} {s.mean():>8.4f} "
              f"{r.std():>8.4f} {s.std():>8.4f} {ks:>7.4f}")
