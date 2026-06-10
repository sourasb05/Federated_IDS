# timevae_toy.py
#
# Standalone toy script:
#   1. Load one CSV from attack_data
#   2. Normalise + build sliding windows  (N, T, F)
#   3. Train TimeVAE — one model per class (0=normal, 1=attack)
#   4. Synthesise samples — class-conditional
#   5. Evaluate: KS test, TSTR/TRTS, distribution plots, correlation heatmaps
#
# Run:
#   python toy/timevae_toy.py

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

# ── make src importable ───────────────────────────────────────
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
from time_vae import train_time_vae, synthesize_time_vae

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
CSV_PATH   = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'attack_data',
    'blackhole_var10_base',
    '10_features_timeseries_60_sec.csv'
)
WINDOW_SIZE  = 10      # T  — seconds per window
N_SYNTH      = 1000    # synthetic samples per class
EPOCHS       = 300
KL_WARMUP    = 100
DEVICE       = 'cpu'
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'timevae_toy_results')
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
# 1. LOAD & NORMALISE
# ═══════════════════════════════════════════════════
print("\n── 1. Loading data ──────────────────────────────")
df = pd.read_csv(CSV_PATH, encoding='utf-8', encoding_errors='ignore')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

feat_cols = [c for c in df.columns if c != 'label']
print(f"   rows={len(df)}  features={len(feat_cols)}  "
      f"label dist: {df['label'].value_counts().to_dict()}")

# min-max normalise using training stats (first 80% rows)
split     = int(len(df) * 0.8)
train_df  = df.iloc[:split].copy()
test_df   = df.iloc[split:].copy()

g_min = train_df[feat_cols].min()
g_max = train_df[feat_cols].max()
denom = (g_max - g_min).replace(0, 1)

for d in [train_df, test_df]:
    d[feat_cols] = (d[feat_cols] - g_min) / denom
    d[feat_cols] = d[feat_cols].clip(0.0, 1.0).fillna(0.0)


# ═══════════════════════════════════════════════════
# 2. SLIDING WINDOWS  (N, T, F)
# ═══════════════════════════════════════════════════
print("\n── 2. Building sliding windows ──────────────────")

def make_windows(df_in: pd.DataFrame, T: int):
    X, y = [], []
    vals   = df_in[feat_cols].values.astype(np.float32)
    labels = df_in['label'].values.astype(int)
    for i in range(len(vals) - T):
        X.append(vals[i:i+T])
        y.append(labels[i + T - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X_train, y_train = make_windows(train_df, WINDOW_SIZE)
X_test,  y_test  = make_windows(test_df,  WINDOW_SIZE)

print(f"   Train: X={X_train.shape}  y dist={dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"   Test : X={X_test.shape}   y dist={dict(zip(*np.unique(y_test,  return_counts=True)))}")

N, T, F = X_train.shape


# ═══════════════════════════════════════════════════
# 3. COMPUTE TIMEVAE CONFIG
# ═══════════════════════════════════════════════════
latent_dim = max(4, min((T * F) // 10, 32))
hidden_dim = max(64, T * F * 4)
free_bits  = round(min(0.5, 8.0 / latent_dim), 4)
batch_size = min(256, max(32, N // 20))

print(f"\n── 3. TimeVAE config ────────────────────────────")
print(f"   T={T}  F={F}  latent={latent_dim}  "
      f"hidden={hidden_dim}  free_bits={free_bits}  batch={batch_size}")


# ═══════════════════════════════════════════════════
# 4. TRAIN ONE TIMEVAE PER CLASS
# ═══════════════════════════════════════════════════
print("\n── 4. Training TimeVAE (class-conditional) ──────")
generators = {}
for cls in [0, 1]:
    X_cls = X_train[y_train == cls]
    print(f"\n   Class {cls}: {len(X_cls)} windows")
    if len(X_cls) < batch_size:
        print(f"   [skip] not enough samples for class {cls}")
        continue
    model = train_time_vae(
        X_np         = X_cls,
        T            = T,
        F            = F,
        latent_dim   = latent_dim,
        hidden_dim   = hidden_dim,
        enc_filters  = (32, 64),
        kernel_size  = 3,
        epochs       = EPOCHS,
        batch_size   = batch_size,
        lr           = 1e-3,
        recon_weight = 2.0,
        kl_warmup    = KL_WARMUP,
        free_bits    = free_bits,
        device       = DEVICE,
    )
    generators[cls] = model
    print(f"   Class {cls} generator trained ✓")


# ═══════════════════════════════════════════════════
# 5. SYNTHESISE
# ═══════════════════════════════════════════════════
print("\n── 5. Synthesising ──────────────────────────────")
syn_parts_X, syn_parts_y = [], []
for cls, model in generators.items():
    X_syn = synthesize_time_vae(model, N_SYNTH, device=DEVICE)  # (N_SYNTH, T, F)
    syn_parts_X.append(X_syn)
    syn_parts_y.append(np.full(N_SYNTH, cls, dtype=np.int64))
    print(f"   Class {cls}: synthesised {N_SYNTH} windows  "
          f"min={X_syn.min():.3f}  max={X_syn.max():.3f}")

X_syn_all = np.concatenate(syn_parts_X, axis=0)   # (2*N_SYNTH, T, F)
y_syn_all = np.concatenate(syn_parts_y, axis=0)


# ═══════════════════════════════════════════════════
# 6. FLATTEN for evaluation  (N, F) — mean over T
# ═══════════════════════════════════════════════════
# Compare per-feature marginals by averaging across time steps.
X_real_flat = X_train.mean(axis=1)       # (N_train, F)
X_syn_flat  = X_syn_all.mean(axis=1)     # (2*N_SYNTH, F)
X_test_flat = X_test.mean(axis=1)        # (N_test, F)

col_names = feat_cols   # F original feature names
real_df_eval = pd.DataFrame(X_real_flat, columns=col_names)
syn_df_eval  = pd.DataFrame(X_syn_flat,  columns=col_names)


# ═══════════════════════════════════════════════════
# 7. KS TEST per feature
# ═══════════════════════════════════════════════════
print("\n── 6. KS Test (per feature) ─────────────────────")
ks_rows = []
for col in col_names:
    r = real_df_eval[col].values
    s = syn_df_eval[col].values
    stat, p = stats.ks_2samp(r, s)
    ks_rows.append({'feature': col,
                    'ks_stat': round(float(stat), 4),
                    'p_value': round(float(p),    4),
                    'similar': bool(p > 0.05)})

ks_df = pd.DataFrame(ks_rows).sort_values('ks_stat', ascending=False)
print(ks_df.to_string(index=False))
print(f"\n   Mean KS  : {ks_df['ks_stat'].mean():.4f}")
print(f"   Similar  : {ks_df['similar'].sum()}/{len(ks_df)} features pass (p>0.05)")
ks_df.to_csv(os.path.join(OUT_DIR, 'ks_results.csv'), index=False)


# ═══════════════════════════════════════════════════
# 8. TSTR / TRTS
# ═══════════════════════════════════════════════════
print("\n── 7. TSTR / TRTS ───────────────────────────────")

def clf_eval(X_tr, y_tr, X_te, y_te, tag):
    clf = LogisticRegression(max_iter=1000, C=1.0,
                              solver='lbfgs', random_state=42)
    try:
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        acc = float(accuracy_score(y_te, preds))
        f1  = float(f1_score(y_te, preds, average='macro', zero_division=0))
    except Exception as e:
        print(f"   [{tag}] failed: {e}")
        acc, f1 = 0.0, 0.0
    quality = 'GOOD' if f1 > 0.60 else ('OK' if f1 > 0.40 else 'POOR')
    print(f"   {tag:30s}  acc={acc:.4f}  f1={f1:.4f}  {quality}")
    return acc, f1

# TSTR: train on synthetic, test on real
tstr_acc, tstr_f1 = clf_eval(X_syn_flat, y_syn_all,
                              X_test_flat, y_test,
                              'TSTR (syn→real)')
# TRTS: train on real, test on synthetic
trts_acc, trts_f1 = clf_eval(X_real_flat, y_train,
                              X_syn_flat,  y_syn_all,
                              'TRTS (real→syn)')
# Baseline: train on real, test on real
base_acc, base_f1 = clf_eval(X_real_flat, y_train,
                              X_test_flat,  y_test,
                              'Baseline (real→real)')

print(f"\n   Gap TRTS-TSTR f1 = {trts_f1 - tstr_f1:+.4f}  "
      f"(0 = perfect match, large positive = distribution mismatch)")


# ═══════════════════════════════════════════════════
# 9. PLOTS
# ═══════════════════════════════════════════════════
print("\n── 8. Saving plots ──────────────────────────────")

# ── 9a. Distribution overlays (all features) ──────
ks_lookup = ks_df.set_index('feature')
n_feat  = len(col_names)
ncols   = min(7, n_feat)
nrows   = (n_feat + ncols - 1) // ncols
cols_sorted = ks_df['feature'].tolist()   # sorted worst KS first

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(3.5*ncols, 3.0*nrows),
                          constrained_layout=True)
axes_flat = np.array(axes).flatten() if n_feat > 1 else [axes]

for ax, col in zip(axes_flat, cols_sorted):
    r = real_df_eval[col].values
    s = syn_df_eval[col].values
    ax.hist(r, bins=30, alpha=0.35, color='steelblue',
            density=True, label='Real')
    ax.hist(s, bins=30, alpha=0.35, color='darkorange',
            density=True, label='Synthetic')
    try:
        xs = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 200)
        ax.plot(xs, stats.gaussian_kde(r)(xs), color='steelblue',  lw=1.5)
        ax.plot(xs, stats.gaussian_kde(s)(xs), color='darkorange', lw=1.5)
    except Exception:
        pass
    ks_v = ks_lookup.loc[col, 'ks_stat']
    p_v  = ks_lookup.loc[col, 'p_value']
    ax.set_title(f"{col}\nKS={ks_v:.3f}  p={p_v:.3f}", fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=5)

for ax in axes_flat[n_feat:]:
    ax.set_visible(False)

n_pass = int(ks_df['similar'].sum())
fig.suptitle(
    f'Real vs Synthetic — TimeVAE  |  window={WINDOW_SIZE}s  '
    f'latent={latent_dim}\n'
    f'{n_pass}/{n_feat} features pass KS (p>0.05)  |  '
    f'TSTR f1={tstr_f1:.3f}  TRTS f1={trts_f1:.3f}',
    fontsize=10
)
path = os.path.join(OUT_DIR, 'distributions.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")

# ── 9b. KS statistic bar chart ─────────────────────
fig, ax = plt.subplots(figsize=(max(8, n_feat * 0.6), 4),
                        constrained_layout=True)
colors = ['tomato' if not r else 'steelblue'
          for r in ks_df['similar'].tolist()]
ax.bar(ks_df['feature'], ks_df['ks_stat'], color=colors)
ax.axhline(0.05, color='black', linestyle='--', lw=1.2, label='KS=0.05')
ax.set_xlabel('Feature')
ax.set_ylabel('KS Statistic (lower = better)')
ax.set_title('KS Statistic per Feature\n'
             'Blue = passes (p>0.05)   Red = fails')
ax.legend()
plt.xticks(rotation=45, ha='right', fontsize=8)
path = os.path.join(OUT_DIR, 'ks_bar.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")

# ── 9c. Correlation heatmaps ───────────────────────
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5),
                                   constrained_layout=True)
rc = real_df_eval.corr()
sc = syn_df_eval.corr()
dc = (rc - sc).abs()
kw = dict(cmap='coolwarm', vmin=-1, vmax=1,
          square=True, xticklabels=True, yticklabels=True,
          annot=True, fmt='.2f', annot_kws={'size': 6})
sns.heatmap(rc, ax=a1, **kw);  a1.set_title('Real correlation')
sns.heatmap(sc, ax=a2, **kw);  a2.set_title('Synthetic correlation')
sns.heatmap(dc, ax=a3, cmap='Reds', vmin=0, vmax=1,
            square=True, xticklabels=True, yticklabels=True,
            annot=True, fmt='.2f', annot_kws={'size': 6})
a3.set_title('|Difference|  (lower = better)')
fig.suptitle('Correlation Structure — Real vs Synthetic', fontsize=12)
path = os.path.join(OUT_DIR, 'correlations.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")

# ── 9d. TSTR/TRTS summary bar ──────────────────────
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
labels_bar = ['Baseline\n(real→real)', 'TRTS\n(real→syn)', 'TSTR\n(syn→real)']
f1_vals    = [base_f1, trts_f1, tstr_f1]
colors_bar = ['steelblue', 'seagreen', 'darkorange']
bars = ax.bar(labels_bar, f1_vals, color=colors_bar, width=0.5)
for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Macro F1')
ax.set_title('TSTR / TRTS / Baseline\n'
             'TSTR → TRTS gap measures distribution mismatch')
ax.axhline(0.5, color='red', linestyle='--', lw=1, label='random (0.5)')
ax.legend()
path = os.path.join(OUT_DIR, 'tstr_trts.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")

# ── 9e. Class-conditional distribution overlay ─────
# Side-by-side: real class 0 vs syn class 0 and real class 1 vs syn class 1
fig, axes = plt.subplots(2, n_feat,
                          figsize=(2.5*n_feat, 6),
                          constrained_layout=True)
for row, cls in enumerate([0, 1]):
    real_cls = X_real_flat[y_train == cls]
    syn_cls  = X_syn_flat[y_syn_all == cls]
    for col_idx, col in enumerate(col_names):
        ax = axes[row, col_idx]
        r = real_cls[:, col_idx]
        s = syn_cls[:, col_idx]
        ax.hist(r, bins=20, alpha=0.4, color='steelblue',
                density=True, label='Real')
        ax.hist(s, bins=20, alpha=0.4, color='darkorange',
                density=True, label='Syn')
        try:
            xs = np.linspace(min(r.min(), s.min()),
                             max(r.max(), s.max()), 150)
            ax.plot(xs, stats.gaussian_kde(r)(xs),
                    color='steelblue', lw=1.2)
            ax.plot(xs, stats.gaussian_kde(s)(xs),
                    color='darkorange', lw=1.2)
        except Exception:
            pass
        ax.set_title(col, fontsize=6)
        ax.tick_params(labelsize=5)
        if col_idx == 0:
            ax.set_ylabel(f'Class {cls}', fontsize=8)
        if row == 0 and col_idx == 0:
            ax.legend(fontsize=6)

fig.suptitle('Class-Conditional Distributions — Real vs Synthetic',
             fontsize=11)
path = os.path.join(OUT_DIR, 'class_conditional.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")

# ═══════════════════════════════════════════════════
# 10. SUMMARY
# ═══════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════")
print("  SUMMARY")
print("══════════════════════════════════════════════════")
print(f"  Window size      : {WINDOW_SIZE}  (T)")
print(f"  Features         : {F}            (F)")
print(f"  Latent dim       : {latent_dim}")
print(f"  Train windows    : {N}")
print(f"  Synth per class  : {N_SYNTH}")
print(f"  Mean KS stat     : {ks_df['ks_stat'].mean():.4f}  "
      f"({'GOOD' if ks_df['ks_stat'].mean()<0.05 else 'OK' if ks_df['ks_stat'].mean()<0.15 else 'POOR'})")
print(f"  KS pass rate     : {ks_df['similar'].sum()}/{len(ks_df)}")
print(f"  Baseline f1      : {base_f1:.4f}  (real→real ceiling)")
print(f"  TRTS f1          : {trts_f1:.4f}  (real→syn)")
print(f"  TSTR f1          : {tstr_f1:.4f}  (syn→real)")
print(f"  TSTR/TRTS gap    : {abs(trts_f1-tstr_f1):.4f}  (lower = better)")
print(f"\n  Results saved to : {OUT_DIR}")
print("══════════════════════════════════════════════════\n")
