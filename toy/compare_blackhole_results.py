# compare_blackhole_results.py
#
# Side-by-side comparison of two Blackhole synthesis approaches:
#
#   AR(1)+ZICVAE  (blackhole_perturb_results_old)
#     — rank level preserved (no differencing)
#     — one conditional ZICVAE for background, AR(1) perturbation for rank
#
#   ZIRVAE        (blackhole_perturb_results)
#     — rank differenced (sign-log1p), separate per-class models
#     — no AR(1); VAE must learn rank from temporal change pattern
#
# Run:
#   conda run -n vinnova python toy/compare_blackhole_results.py
#
# Output:
#   toy/blackhole_comparison/comparison_table.csv
#   toy/blackhole_comparison/tstr_comparison.png
#   toy/blackhole_comparison/ks_comparison.png
#   toy/blackhole_comparison/cond_precision_comparison.png
#   toy/blackhole_comparison/radar_comparison.png

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
OLD_DIR = os.path.join(BASE, 'blackhole_perturb_results_old')   # AR(1)+ZICVAE
NEW_DIR = os.path.join(BASE, 'blackhole_perturb_results')       # ZIRVAE
OUT_DIR = os.path.join(BASE, 'blackhole_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

VARIANTS = sorted([d for d in os.listdir(OLD_DIR) if d.startswith('blackhole_')])

METRICS = ['baseline_f1', 'rank_only_f1', 'tstr_f1', 'trts_f1',
           'cond_precision', 'ks_cls0_mean', 'ks_cls1_mean', 'mean_tail_ks']

LABELS = {
    'baseline_f1':    'Baseline F1',
    'rank_only_f1':   'Rank-only F1',
    'tstr_f1':        'TSTR F1',
    'trts_f1':        'TRTS F1',
    'cond_precision': 'Cond. Precision',
    'ks_cls0_mean':   'KS Normal (↓ better)',
    'ks_cls1_mean':   'KS Attack (↓ better)',
    'mean_tail_ks':   'Tail KS (↓ better)',
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(root, json_name):
    rows = []
    for v in VARIANTS:
        p = os.path.join(root, v, json_name)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        row = {'variant': v}
        for m in METRICS:
            row[m] = d.get(m, None)
        rows.append(row)
    return pd.DataFrame(rows)

df_ar1   = load_metrics(OLD_DIR, 'metrics.json')
df_zi    = load_metrics(NEW_DIR, 'zirvae_metrics.json')

df_ar1['method'] = 'AR(1)+ZICVAE'
df_zi['method']  = 'ZIRVAE'

# Short variant labels: blackhole_var10_base → v10_base
def short(v):
    return v.replace('blackhole_', 'bh_')

df_ar1['var_short'] = df_ar1['variant'].apply(short)
df_zi['var_short']  = df_zi['variant'].apply(short)

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────────────────────

print('\n' + '='*110)
print('  BLACKHOLE SYNTHESIS COMPARISON  —  AR(1)+ZICVAE  vs  ZIRVAE')
print('='*110)

hdr = (f"  {'Variant':<22} {'Method':<14} "
       f"{'Base':>6} {'RkOnly':>7} {'TSTR':>6} {'TRTS':>6} "
       f"{'CondP':>6} {'KS0':>6} {'KS1':>6} {'TailKS':>7}")
print(hdr)
print('-' * len(hdr))

rows_merged = []
for v in VARIANTS:
    vs = short(v)
    r_ar1 = df_ar1[df_ar1['variant'] == v]
    r_zi  = df_zi[df_zi['variant']  == v]

    for label, df_m in [('AR1+CVAE', r_ar1), ('ZIRVAE', r_zi)]:
        if df_m.empty:
            continue
        row = df_m.iloc[0]
        tail = f"{row['mean_tail_ks']:.3f}" if row['mean_tail_ks'] is not None else '  N/A'
        print(f"  {vs:<22} {label:<14} "
              f"{row['baseline_f1']:>6.3f} "
              f"{row['rank_only_f1']:>7.3f} "
              f"{row['tstr_f1']:>6.3f} "
              f"{row['trts_f1']:>6.3f} "
              f"{row['cond_precision']:>6.3f} "
              f"{row['ks_cls0_mean']:>6.3f} "
              f"{row['ks_cls1_mean']:>6.3f} "
              f"{tail:>7}")
        rows_merged.append(row.to_dict())
    print()

# ─────────────────────────────────────────────────────────────────────────────
# DELTA TABLE  (ZIRVAE − AR1+ZICVAE, positive = ZIRVAE better for TSTR/TRTS/CondP,
#                                     negative = ZIRVAE better for KS)
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*110)
print('  DELTA TABLE  (ZIRVAE − AR(1)+ZICVAE)')
print('  Positive = ZIRVAE better for: TSTR, TRTS, Cond.Precision')
print('  Negative = ZIRVAE better for: KS0, KS1, TailKS')
print('='*110)
print(hdr.replace('Method', 'Δ ZIRVAE−AR1'))
print('-' * len(hdr))

delta_rows = []
for v in VARIANTS:
    vs    = short(v)
    r_ar1 = df_ar1[df_ar1['variant'] == v]
    r_zi  = df_zi[df_zi['variant']  == v]
    if r_ar1.empty or r_zi.empty:
        continue
    a = r_ar1.iloc[0]
    z = r_zi.iloc[0]

    d_tstr  = z['tstr_f1']       - a['tstr_f1']
    d_trts  = z['trts_f1']       - a['trts_f1']
    d_ks0   = z['ks_cls0_mean']  - a['ks_cls0_mean']
    d_ks1   = z['ks_cls1_mean']  - a['ks_cls1_mean']
    d_cp    = z['cond_precision'] - a['cond_precision']
    d_rk    = z['rank_only_f1']  - a['rank_only_f1']

    a_tail  = a['mean_tail_ks'] if a['mean_tail_ks'] is not None else float('nan')
    z_tail  = z['mean_tail_ks'] if z['mean_tail_ks'] is not None else float('nan')
    d_tail  = z_tail - a_tail

    tail_s = f"{d_tail:>+7.3f}" if np.isfinite(d_tail) else '    N/A'
    print(f"  {vs:<22} {'':14} "
          f"{'---':>6} "
          f"{d_rk:>+7.3f} "
          f"{d_tstr:>+6.3f} "
          f"{d_trts:>+6.3f} "
          f"{d_cp:>+6.3f} "
          f"{d_ks0:>+6.3f} "
          f"{d_ks1:>+6.3f} "
          f"{tail_s}")

    delta_rows.append({
        'variant': v, 'var_short': vs,
        'd_tstr': d_tstr, 'd_trts': d_trts,
        'd_cond_precision': d_cp, 'd_rank_only_f1': d_rk,
        'd_ks0': d_ks0, 'd_ks1': d_ks1, 'd_tail_ks': d_tail,
    })

df_delta = pd.DataFrame(delta_rows)

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*80)
print('  AGGREGATE  (mean ± std across 12 variants)')
print('='*80)
agg_cols = ['tstr_f1', 'trts_f1', 'cond_precision',
            'ks_cls0_mean', 'ks_cls1_mean', 'rank_only_f1']
agg_lbl  = ['TSTR', 'TRTS', 'Cond.Prec', 'KS-Normal', 'KS-Attack', 'RankOnly']

hdr2 = f"  {'Metric':<18} {'AR(1)+ZICVAE':>14} {'ZIRVAE':>14} {'Δ (ZI−AR1)':>14}"
print(hdr2); print('-' * len(hdr2))
for col, lbl in zip(agg_cols, agg_lbl):
    a_vals = df_ar1[col].dropna()
    z_vals = df_zi[col].dropna()
    a_m, a_s = a_vals.mean(), a_vals.std()
    z_m, z_s = z_vals.mean(), z_vals.std()
    delta    = z_m - a_m
    sign     = '+' if delta >= 0 else ''
    print(f"  {lbl:<18} {a_m:>7.3f} ± {a_s:.3f}  {z_m:>7.3f} ± {z_s:.3f}  "
          f"  {sign}{delta:>+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────
df_ar1_out = df_ar1.copy(); df_ar1_out['method'] = 'AR1_ZICVAE'
df_zi_out  = df_zi.copy();  df_zi_out['method']  = 'ZIRVAE'
combined   = pd.concat([df_ar1_out, df_zi_out], ignore_index=True)
combined.to_csv(os.path.join(OUT_DIR, 'comparison_table.csv'), index=False)
df_delta.to_csv(os.path.join(OUT_DIR, 'delta_table.csv'), index=False)
print(f"\n  Saved CSVs → {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — TSTR / TRTS / Cond.Precision grouped bars
# ─────────────────────────────────────────────────────────────────────────────
vs_labels = df_ar1['var_short'].tolist()
x         = np.arange(len(vs_labels))
w         = 0.18

fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)

for ax, (metric, title) in zip(axes, [
    ('tstr_f1',        'TSTR F1  (Train Syn → Test Real)'),
    ('trts_f1',        'TRTS F1  (Train Real → Test Syn)'),
    ('cond_precision', 'Cond. Precision  (real clf on syn attacks)'),
]):
    a_vals = df_ar1.set_index('variant').reindex(
        df_ar1['variant'])[metric].values
    z_vals = df_zi.set_index('variant').reindex(
        df_zi['variant'])[metric].values
    b_vals = df_ar1['baseline_f1'].values   # same baseline for both

    ax.bar(x - w,   b_vals, w, label='Baseline',      color='#4C72B0', alpha=0.75)
    ax.bar(x,       a_vals, w, label='AR(1)+ZICVAE',  color='#DD8452', alpha=0.85)
    ax.bar(x + w,   z_vals, w, label='ZIRVAE',        color='#55A868', alpha=0.85)

    for i, (bv, av, zv) in enumerate(zip(b_vals, a_vals, z_vals)):
        for off, val in [(-w, bv), (0, av), (w, zv)]:
            ax.text(i + off, val + 0.01, f'{val:.2f}',
                    ha='center', fontsize=5.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(vs_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel('Macro F1')
    ax.axhline(0.5, color='red', linestyle='--', lw=0.8, label='random')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)

fig.suptitle('AR(1)+ZICVAE  vs  ZIRVAE — Blackhole Variants', fontsize=12, fontweight='bold')
path = os.path.join(OUT_DIR, 'tstr_comparison.png')
fig.savefig(path, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — KS per class (lower = better)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

for ax, (metric, title) in zip(axes, [
    ('ks_cls0_mean', 'Mean KS — Normal class  (↓ better)'),
    ('ks_cls1_mean', 'Mean KS — Attack class  (↓ better)'),
]):
    a_vals = df_ar1[metric].values
    z_vals = df_zi[metric].values

    ax.bar(x - w/2, a_vals, w, label='AR(1)+ZICVAE', color='#DD8452', alpha=0.85)
    ax.bar(x + w/2, z_vals, w, label='ZIRVAE',        color='#55A868', alpha=0.85)

    for i, (av, zv) in enumerate(zip(a_vals, z_vals)):
        ax.text(i - w/2, av + 0.003, f'{av:.3f}', ha='center', fontsize=6)
        ax.text(i + w/2, zv + 0.003, f'{zv:.3f}', ha='center', fontsize=6)

    ax.axhline(0.05, color='black', linestyle='--', lw=1, label='p=0.05 threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(vs_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel('Mean KS Statistic')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

fig.suptitle('Distribution Fidelity (KS) — AR(1)+ZICVAE  vs  ZIRVAE', fontsize=12, fontweight='bold')
path = os.path.join(OUT_DIR, 'ks_comparison.png')
fig.savefig(path, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Delta heatmap  (ZIRVAE − AR1+ZICVAE)
# ─────────────────────────────────────────────────────────────────────────────
delta_plot_cols = ['d_tstr', 'd_trts', 'd_cond_precision',
                   'd_ks0',  'd_ks1',  'd_rank_only_f1']
delta_plot_lbl  = ['ΔTSTR', 'ΔTRTS', 'ΔCondPrec', 'ΔKS-Normal', 'ΔKS-Attack', 'ΔRankOnly']

# For KS metrics, flip sign so green always = ZIRVAE better
heat_data = df_delta[['var_short'] + delta_plot_cols].set_index('var_short').copy()
heat_data_display = heat_data.copy()
heat_data_display['d_ks0'] = -heat_data_display['d_ks0']
heat_data_display['d_ks1'] = -heat_data_display['d_ks1']
heat_data_display.columns = ['ΔTSTR', 'ΔTRTS', 'ΔCondPrec',
                              'ΔKS-Normal\n(−=better)', 'ΔKS-Attack\n(−=better)',
                              'ΔRankOnly']

# Rename columns for display: flip columns renamed above
heat_arr = heat_data_display.values.astype(float)

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
im = ax.imshow(heat_arr, cmap='RdYlGn', aspect='auto', vmin=-0.35, vmax=0.35)

ax.set_xticks(range(len(heat_data_display.columns)))
ax.set_xticklabels(heat_data_display.columns, fontsize=9)
ax.set_yticks(range(len(heat_data_display.index)))
ax.set_yticklabels(heat_data_display.index, fontsize=8)

for i in range(heat_arr.shape[0]):
    for j in range(heat_arr.shape[1]):
        v = heat_arr[i, j]
        txt = f'{v:+.3f}' if np.isfinite(v) else 'N/A'
        color = 'black' if abs(v) < 0.25 else 'white'
        ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, ax=ax, label='Δ value  (green = ZIRVAE better)')
ax.set_title('Delta Heatmap: ZIRVAE − AR(1)+ZICVAE\n'
             'Green = ZIRVAE wins  |  Red = AR(1)+ZICVAE wins', fontsize=11)
path = os.path.join(OUT_DIR, 'delta_heatmap.png')
fig.savefig(path, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Scatter: TSTR AR1 vs TSTR ZIRVAE (one dot per variant)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

for ax, (metric, title) in zip(axes, [
    ('tstr_f1', 'TSTR F1'),
    ('trts_f1', 'TRTS F1'),
]):
    a_vals = df_ar1[metric].values
    z_vals = df_zi[metric].values
    lim    = (0.2, 1.0)

    ax.scatter(a_vals, z_vals, s=60, zorder=3, color='steelblue', edgecolors='k', lw=0.5)
    for i, vs in enumerate(vs_labels):
        ax.annotate(vs, (a_vals[i], z_vals[i]),
                    textcoords='offset points', xytext=(4, 2), fontsize=6)

    ax.plot(lim, lim, 'r--', lw=1, label='y = x  (equal)')
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel(f'AR(1)+ZICVAE  {title}', fontsize=9)
    ax.set_ylabel(f'ZIRVAE  {title}', fontsize=9)
    ax.set_title(f'{title} — scatter per variant\n(above diagonal = ZIRVAE better)',
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle('AR(1)+ZICVAE  vs  ZIRVAE — per-variant scatter', fontsize=11, fontweight='bold')
path = os.path.join(OUT_DIR, 'scatter_comparison.png')
fig.savefig(path, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Win/Tie/Loss count bar
# ─────────────────────────────────────────────────────────────────────────────
# For each metric, count variants where AR1+ZICVAE wins, ZIRVAE wins, or tie (|Δ| < 0.01)
compare_metrics = {
    'tstr_f1':        ('TSTR',        'higher'),
    'trts_f1':        ('TRTS',        'higher'),
    'cond_precision': ('Cond.Prec',   'higher'),
    'ks_cls0_mean':   ('KS-Normal',   'lower'),
    'ks_cls1_mean':   ('KS-Attack',   'lower'),
}

win_ar1, win_zi, tie_counts = [], [], []
metric_names = []

for metric, (lbl, direction) in compare_metrics.items():
    a_vals = df_ar1[metric].values
    z_vals = df_zi[metric].values
    w_ar1 = w_zi = w_tie = 0
    for a, z in zip(a_vals, z_vals):
        delta = z - a
        if abs(delta) < 0.01:
            w_tie += 1
        elif (direction == 'higher' and delta > 0) or (direction == 'lower' and delta < 0):
            w_zi += 1
        else:
            w_ar1 += 1
    win_ar1.append(w_ar1)
    win_zi.append(w_zi)
    tie_counts.append(w_tie)
    metric_names.append(lbl)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
xm = np.arange(len(metric_names))
wm = 0.25
ax.bar(xm - wm, win_ar1,    wm, label='AR(1)+ZICVAE wins', color='#DD8452', alpha=0.85)
ax.bar(xm,      win_zi,     wm, label='ZIRVAE wins',        color='#55A868', alpha=0.85)
ax.bar(xm + wm, tie_counts, wm, label='Tie (|Δ|<0.01)',     color='#8172B2', alpha=0.60)
for i, (a, z, t) in enumerate(zip(win_ar1, win_zi, tie_counts)):
    ax.text(i - wm, a + 0.05, str(a), ha='center', fontsize=9)
    ax.text(i,      z + 0.05, str(z), ha='center', fontsize=9)
    ax.text(i + wm, t + 0.05, str(t), ha='center', fontsize=9)
ax.set_xticks(xm)
ax.set_xticklabels(metric_names, fontsize=10)
ax.set_ylabel('Number of variants (out of 12)')
ax.set_ylim(0, 14)
ax.set_title('Win/Tie/Loss count per metric  (across 12 variants)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
path = os.path.join(OUT_DIR, 'win_loss_count.png')
fig.savefig(path, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path}")

print(f"\n  All outputs → {OUT_DIR}\n")
