# feature_distribution_analysis.py
#
# Analyse per-feature distributions for label=0 (normal) vs label=1 (attack)
# across all 48 attack folders in attack_data/.
#
# For each folder:
#   - Loads all 20 CSV files and concatenates them
#   - Separates rows by label
#   - Computes descriptive stats (mean, std, skew, kurtosis, zero%)
#   - Auto-detects distribution type using the same thresholds as zirvae_multifile.py
#     ('bernoulli', 'zi_lognorm', 'lognormal', 'continuous')
#   - Runs KS test between normal and attack for each feature
#   - Plots distribution overlays (PDF + CDF)
#   - Saves per-folder metrics.json and a global summary CSV
#
# Run:  conda run -n vinnova python toy/feature_distribution_analysis.py

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, 'attack_data')
OUT_ROOT  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'feature_dist_results')
os.makedirs(OUT_ROOT, exist_ok=True)

# ── distribution-type thresholds (mirror zirvae_multifile.py) ────────────────
BERNOULLI_ZERO_THRESH = 0.90
ZI_ZERO_THRESH        = 0.30
LOGNORMAL_SKEW_THRESH = 1.0
RANK_COLS             = ['rank', 'rank.1']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_folder(folder_path: str) -> pd.DataFrame:
    """Concatenate all CSV files in a folder into one DataFrame."""
    dfs = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(folder_path, fname),
                         encoding='utf-8', encoding_errors='ignore')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def detect_type(x: np.ndarray, col_name: str) -> str:
    """
    Mirror of detect_feature_types() in zirvae_multifile.py.
    Returns one of: 'bernoulli', 'zi_lognorm', 'lognormal', 'continuous'
    """
    if col_name in RANK_COLS:
        return 'continuous'
    n_unique = len(np.unique(x))
    zero_pct = float((x == 0).mean())
    col_skew = float(stats.skew(x))
    if zero_pct >= BERNOULLI_ZERO_THRESH and n_unique <= 3:
        return 'bernoulli'
    if zero_pct >= ZI_ZERO_THRESH:
        return 'zi_lognorm'
    if x.min() >= 0 and col_skew > LOGNORMAL_SKEW_THRESH:
        return 'lognormal'
    return 'continuous'


def feature_stats(x: np.ndarray) -> dict:
    """Return descriptive stats for a 1-D array."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    return {
        'n':        int(len(x)),
        'mean':     float(np.mean(x)),
        'std':      float(np.std(x)),
        'median':   float(np.median(x)),
        'min':      float(np.min(x)),
        'max':      float(np.max(x)),
        'skew':     float(stats.skew(x)),
        'kurtosis': float(stats.kurtosis(x)),
        'zero_pct': float((x == 0).mean()),
    }


def ks_test(x0: np.ndarray, x1: np.ndarray):
    """KS two-sample test between normal and attack distributions."""
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]
    if len(x0) < 5 or len(x1) < 5:
        return float('nan'), float('nan')
    res = stats.ks_2samp(x0, x1)
    return float(res.statistic), float(res.pvalue)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(df_normal: pd.DataFrame,
                       df_attack: pd.DataFrame,
                       feat_cols: list,
                       folder_name: str,
                       out_dir: str):
    """
    One figure per feature: left = PDF overlay (KDE), right = ECDF overlay.
    """
    n_feat = len(feat_cols)
    n_cols = 2
    n_rows = n_feat

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(12, 3 * n_rows),
                             squeeze=False)
    fig.suptitle(f'Feature Distributions — {folder_name}\n'
                 f'Normal (n={len(df_normal):,})  vs  Attack (n={len(df_attack):,})',
                 fontsize=14, y=1.001)

    colors = {'normal': '#2196F3', 'attack': '#F44336'}

    for row, col in enumerate(feat_cols):
        x0 = df_normal[col].dropna().values
        x1 = df_attack[col].dropna().values if col in df_attack.columns else np.array([])

        ax_pdf = axes[row, 0]
        ax_cdf = axes[row, 1]

        # ── PDF / KDE ────────────────────────────────────────────────────
        if len(x0) > 1:
            try:
                sns.kdeplot(x0, ax=ax_pdf, label='Normal',
                            color=colors['normal'], fill=True, alpha=0.35, linewidth=1.5)
            except Exception:
                ax_pdf.hist(x0, bins=30, density=True, alpha=0.4,
                            color=colors['normal'], label='Normal')
        if len(x1) > 1:
            try:
                sns.kdeplot(x1, ax=ax_pdf, label='Attack',
                            color=colors['attack'], fill=True, alpha=0.35, linewidth=1.5)
            except Exception:
                ax_pdf.hist(x1, bins=30, density=True, alpha=0.4,
                            color=colors['attack'], label='Attack')

        ax_pdf.set_ylabel(col, fontsize=8, rotation=0, labelpad=60, va='center')
        ax_pdf.set_xlabel('')
        if row == 0:
            ax_pdf.set_title('PDF (KDE)')
        ax_pdf.legend(fontsize=7, loc='upper right')
        ax_pdf.tick_params(labelsize=7)

        # ── ECDF ─────────────────────────────────────────────────────────
        for xarr, lbl, clr in [(x0, 'Normal', colors['normal']),
                                 (x1, 'Attack', colors['attack'])]:
            if len(xarr) > 0:
                xs = np.sort(xarr)
                ys = np.arange(1, len(xs) + 1) / len(xs)
                ax_cdf.step(xs, ys, label=lbl, color=clr, linewidth=1.5)

        if row == 0:
            ax_cdf.set_title('ECDF')
        ax_cdf.legend(fontsize=7, loc='lower right')
        ax_cdf.set_xlabel('')
        ax_cdf.set_ylabel('')
        ax_cdf.tick_params(labelsize=7)

        # KS annotation
        ks_stat, ks_p = ks_test(x0, x1)
        ax_cdf.annotate(f'KS={ks_stat:.3f}, p={ks_p:.3f}',
                        xy=(0.02, 0.97), xycoords='axes fraction',
                        va='top', ha='left', fontsize=7,
                        color='green' if ks_p < 0.05 else 'gray')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'distributions_{folder_name}.png')
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_dist_type_heatmap(summary_df: pd.DataFrame, out_dir: str):
    """
    Heatmap: rows = attack folders, cols = features,
    cell = distribution type (one colour per type).
    Separate plots per attack family.
    """
    type_map  = {'bernoulli': 0, 'zi_lognorm': 1, 'lognormal': 2, 'continuous': 3}
    type_labels = {0: 'bernoulli', 1: 'zi_lognorm', 2: 'lognormal', 3: 'continuous'}
    cmap      = matplotlib.colors.ListedColormap(['#FF9800', '#9C27B0', '#4CAF50', '#2196F3'])
    bounds    = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm      = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    for label_name in ['normal', 'attack']:
        col_prefix = f'dist_type_{label_name}_'
        type_cols  = [c for c in summary_df.columns if c.startswith(col_prefix)]
        feat_names = [c[len(col_prefix):] for c in type_cols]
        if not type_cols:
            continue

        for attack_family in ['blackhole', 'disflooding', 'localrepair', 'worstparent']:
            sub = summary_df[summary_df['folder'].str.startswith(attack_family)].copy()
            if sub.empty:
                continue
            sub = sub.set_index('folder')[type_cols].rename(
                columns={c: n for c, n in zip(type_cols, feat_names)})
            mat = sub.applymap(lambda v: type_map.get(str(v), -1))

            fig, ax = plt.subplots(figsize=(max(10, len(feat_names)), max(4, len(sub) * 0.5)))
            im = ax.imshow(mat.values, cmap=cmap, norm=norm, aspect='auto')
            ax.set_xticks(range(len(feat_names)))
            ax.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels(sub.index, fontsize=8)
            ax.set_title(f'Distribution Type ({label_name}) — {attack_family}', fontsize=12)

            cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
            cbar.ax.set_yticklabels([type_labels[i] for i in [0, 1, 2, 3]], fontsize=8)

            plt.tight_layout()
            fname = f'dist_type_heatmap_{label_name}_{attack_family}.png'
            fig.savefig(os.path.join(out_dir, fname), dpi=100, bbox_inches='tight')
            plt.close(fig)


def plot_ks_heatmap(summary_df: pd.DataFrame, out_dir: str):
    """
    Heatmap of KS statistics across folders × features.
    One plot per attack family.
    """
    ks_cols    = [c for c in summary_df.columns if c.startswith('ks_stat_')]
    feat_names = [c[len('ks_stat_'):] for c in ks_cols]
    if not ks_cols:
        return

    for attack_family in ['blackhole', 'disflooding', 'localrepair', 'worstparent']:
        sub = summary_df[summary_df['folder'].str.startswith(attack_family)].copy()
        if sub.empty:
            continue
        mat = sub.set_index('folder')[ks_cols].rename(
            columns={c: n for c, n in zip(ks_cols, feat_names)})

        fig, ax = plt.subplots(figsize=(max(10, len(feat_names)), max(4, len(sub) * 0.5)))
        sns.heatmap(mat.astype(float), ax=ax, vmin=0, vmax=1,
                    cmap='RdYlGn_r', annot=True, fmt='.2f',
                    annot_kws={'size': 7}, linewidths=0.3, linecolor='#eee')
        ax.set_title(f'KS Statistic (normal vs attack) — {attack_family}', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        plt.tight_layout()
        fname = f'ks_heatmap_{attack_family}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=100, bbox_inches='tight')
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-folder analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_folder(folder_name: str, folder_path: str, out_dir: str) -> dict:
    """
    Analyse one attack folder.  Returns a flat dict suitable for a summary row.
    """
    print(f'  Analysing {folder_name} …', end=' ', flush=True)
    df = load_folder(folder_path)
    if df.empty or 'label' not in df.columns:
        print('SKIP (empty or no label)')
        return {}

    feat_cols = [c for c in df.columns if c != 'label']
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    df_normal = df[df['label'] == 0][feat_cols]
    df_attack = df[df['label'] == 1][feat_cols]

    n_normal = len(df_normal)
    n_attack = len(df_attack)
    print(f'normal={n_normal:,}, attack={n_attack:,}')

    # Per-feature analysis
    per_feature = {}
    for col in feat_cols:
        x_all    = df[col].dropna().values
        x_normal = df_normal[col].dropna().values
        x_attack = df_attack[col].dropna().values if len(df_attack) > 0 else np.array([])

        dist_type_global = detect_type(x_all,    col)
        dist_type_normal = detect_type(x_normal, col) if len(x_normal) > 0 else 'N/A'
        dist_type_attack = detect_type(x_attack, col) if len(x_attack) > 0 else 'N/A'

        ks_stat, ks_p = ks_test(x_normal, x_attack)

        per_feature[col] = {
            'dist_type_global': dist_type_global,
            'dist_type_normal': dist_type_normal,
            'dist_type_attack': dist_type_attack,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_p,
            'ks_significant': bool(ks_p < 0.05) if not np.isnan(ks_p) else False,
            'normal': feature_stats(x_normal),
            'attack': feature_stats(x_attack),
        }

    # Save per-folder detailed JSON
    os.makedirs(out_dir, exist_ok=True)
    detail = {
        'folder':   folder_name,
        'n_normal': n_normal,
        'n_attack': n_attack,
        'features': per_feature,
    }
    with open(os.path.join(out_dir, 'feature_details.json'), 'w') as fh:
        json.dump(detail, fh, indent=2)

    # Distribution overlay plots
    plot_distributions(df_normal, df_attack, feat_cols, folder_name, out_dir)

    # Build flat summary row
    row = {'folder': folder_name, 'n_normal': n_normal, 'n_attack': n_attack}
    for col, info in per_feature.items():
        row[f'dist_type_global_{col}'] = info['dist_type_global']
        row[f'dist_type_normal_{col}'] = info['dist_type_normal']
        row[f'dist_type_attack_{col}'] = info['dist_type_attack']
        row[f'ks_stat_{col}']          = info['ks_stat']
        row[f'ks_pvalue_{col}']        = info['ks_pvalue']
        row[f'ks_sig_{col}']           = info['ks_significant']
        for split in ('normal', 'attack'):
            for stat_name, stat_val in info[split].items():
                row[f'{split}_{stat_name}_{col}'] = stat_val

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_mean_comparison(summary_df: pd.DataFrame, out_dir: str):
    """
    Bar chart: mean value of each feature for normal vs attack,
    one chart per attack type family (blackhole, disflooding, …).
    Averaged across all variant folders within that family.
    """
    families = ['blackhole', 'disflooding', 'localrepair', 'worstparent']

    # Collect feature columns from summary
    feat_cols_avail = []
    for c in summary_df.columns:
        if c.startswith('normal_mean_'):
            feat_cols_avail.append(c[len('normal_mean_'):])

    if not feat_cols_avail:
        return

    for family in families:
        sub = summary_df[summary_df['folder'].str.startswith(family)]
        if sub.empty:
            continue

        normal_means = sub[[f'normal_mean_{c}' for c in feat_cols_avail]].mean()
        attack_means = sub[[f'attack_mean_{c}' for c in feat_cols_avail]].mean()

        x     = np.arange(len(feat_cols_avail))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(feat_cols_avail)), 5))
        bars0 = ax.bar(x - width/2, normal_means.values, width,
                       label='Normal', color='#2196F3', alpha=0.8)
        bars1 = ax.bar(x + width/2, attack_means.values, width,
                       label='Attack', color='#F44336', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(feat_cols_avail, rotation=45, ha='right', fontsize=9)
        ax.set_title(f'Mean Feature Values — {family} (averaged across variants)', fontsize=12)
        ax.set_ylabel('Mean (raw)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'mean_comparison_{family}.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)


def plot_zero_pct_comparison(summary_df: pd.DataFrame, out_dir: str):
    """
    Heatmap: zero% for each feature × folder, separated by normal vs attack.
    """
    families = ['blackhole', 'disflooding', 'localrepair', 'worstparent']
    feat_cols_avail = [c[len('normal_zero_pct_'):] for c in summary_df.columns
                       if c.startswith('normal_zero_pct_')]
    if not feat_cols_avail:
        return

    for split in ('normal', 'attack'):
        zero_cols = [f'{split}_zero_pct_{c}' for c in feat_cols_avail]
        for family in families:
            sub = summary_df[summary_df['folder'].str.startswith(family)].copy()
            if sub.empty:
                continue
            mat = sub.set_index('folder')[zero_cols].rename(
                columns={c: n for c, n in zip(zero_cols, feat_cols_avail)})
            fig, ax = plt.subplots(figsize=(max(10, len(feat_cols_avail)),
                                            max(4, len(sub) * 0.5)))
            sns.heatmap(mat.astype(float), ax=ax, vmin=0, vmax=1,
                        cmap='YlOrRd', annot=True, fmt='.2f',
                        annot_kws={'size': 7}, linewidths=0.3, linecolor='#eee')
            ax.set_title(f'Zero% ({split}) — {family}', fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            plt.tight_layout()
            fname = f'zero_pct_{split}_{family}.png'
            fig.savefig(os.path.join(out_dir, fname), dpi=100, bbox_inches='tight')
            plt.close(fig)


def plot_skew_comparison(summary_df: pd.DataFrame, out_dir: str):
    """
    Heatmap: skewness of each feature × folder for normal and attack.
    """
    feat_cols_avail = [c[len('normal_skew_'):] for c in summary_df.columns
                       if c.startswith('normal_skew_')]
    if not feat_cols_avail:
        return

    for split in ('normal', 'attack'):
        skew_cols = [f'{split}_skew_{c}' for c in feat_cols_avail]
        for family in ['blackhole', 'disflooding', 'localrepair', 'worstparent']:
            sub = summary_df[summary_df['folder'].str.startswith(family)].copy()
            if sub.empty:
                continue
            mat = sub.set_index('folder')[skew_cols].rename(
                columns={c: n for c, n in zip(skew_cols, feat_cols_avail)})
            fig, ax = plt.subplots(figsize=(max(10, len(feat_cols_avail)),
                                            max(4, len(sub) * 0.5)))
            sns.heatmap(mat.astype(float), ax=ax, center=0,
                        cmap='coolwarm', annot=True, fmt='.2f',
                        annot_kws={'size': 7}, linewidths=0.3, linecolor='#eee')
            ax.set_title(f'Skewness ({split}) — {family}', fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            plt.tight_layout()
            fname = f'skew_{split}_{family}.png'
            fig.savefig(os.path.join(out_dir, fname), dpi=100, bbox_inches='tight')
            plt.close(fig)


def print_dist_type_summary(summary_df: pd.DataFrame):
    """
    Console summary: for each (attack_family, feature) how often each
    distribution type is detected across the 12 variant folders.
    Helps quickly spot which features are consistently ZI or lognormal.
    """
    feat_cols_avail = [c[len('dist_type_global_'):] for c in summary_df.columns
                       if c.startswith('dist_type_global_')]
    print('\n' + '='*80)
    print('DISTRIBUTION TYPE CONSENSUS (global, across all variant folders)')
    print('='*80)
    for family in ['blackhole', 'disflooding', 'localrepair', 'worstparent']:
        sub = summary_df[summary_df['folder'].str.startswith(family)]
        if sub.empty:
            continue
        print(f'\n── {family.upper()} (n_folders={len(sub)}) ──')
        for col in feat_cols_avail:
            counts = sub[f'dist_type_global_{col}'].value_counts()
            dominant = counts.index[0] if len(counts) > 0 else 'N/A'
            detail   = ', '.join(f'{t}:{c}' for t, c in counts.items())
            marker   = '⚠ ' if len(counts) > 1 else '  '
            print(f'  {marker}{col:12s}: {dominant:12s}  [{detail}]')


def print_ks_summary(summary_df: pd.DataFrame):
    """
    Console: per (family, feature) fraction of folders where KS is significant.
    High fraction → feature reliably discriminates normal vs attack.
    """
    feat_cols_avail = [c[len('ks_stat_'):] for c in summary_df.columns
                       if c.startswith('ks_stat_')]
    print('\n' + '='*80)
    print('KS TEST SIGNIFICANCE RATE  (normal vs attack, p<0.05)')
    print('='*80)
    for family in ['blackhole', 'disflooding', 'localrepair', 'worstparent']:
        sub = summary_df[summary_df['folder'].str.startswith(family)]
        if sub.empty:
            continue
        print(f'\n── {family.upper()} ──')
        for col in feat_cols_avail:
            sig_rate = sub[f'ks_sig_{col}'].mean()
            ks_mean  = sub[f'ks_stat_{col}'].mean()
            bar = '█' * int(sig_rate * 20)
            print(f'  {col:12s}: sig_rate={sig_rate:.2f}  mean_KS={ks_mean:.3f}  |{bar:<20}|')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    folders = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ])

    print(f'Found {len(folders)} attack folders under {DATA_ROOT}')
    all_rows = []

    for folder_name in folders:
        folder_path = os.path.join(DATA_ROOT, folder_name)
        folder_out  = os.path.join(OUT_ROOT, folder_name)
        os.makedirs(folder_out, exist_ok=True)
        row = analyse_folder(folder_name, folder_path, folder_out)
        if row:
            all_rows.append(row)

    if not all_rows:
        print('No data processed.')
        return

    summary_df = pd.DataFrame(all_rows)
    summary_csv = os.path.join(OUT_ROOT, 'summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f'\nSummary CSV saved → {summary_csv}')

    # Console summaries
    print_dist_type_summary(summary_df)
    print_ks_summary(summary_df)

    # Aggregate plots
    print('\nGenerating aggregate plots …')
    plot_dist_type_heatmap(summary_df, OUT_ROOT)
    plot_ks_heatmap(summary_df, OUT_ROOT)
    plot_mean_comparison(summary_df, OUT_ROOT)
    plot_zero_pct_comparison(summary_df, OUT_ROOT)
    plot_skew_comparison(summary_df, OUT_ROOT)

    print(f'\nAll outputs saved under {OUT_ROOT}')


if __name__ == '__main__':
    main()
