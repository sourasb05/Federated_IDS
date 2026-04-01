# evaluate_generator.py
#
# Evaluates quality of synthetic data from a trained TVAE.
# Synthetic samples are generated in PCA space and then
# inverse-transformed to original feature space before comparison.
#
# Statistical tests (per feature):
#   - Kolmogorov-Smirnov  (KS-2samp)
#   - Mann-Whitney U      (non-parametric, no normality assumption)
#   - Welch's t-test      (parametric)
#
# Plots:
#   - Distribution overlays  (histogram + KDE, original feature space)
#   - Significance summary   (bar chart of p-values across all features)
#   - Correlation heatmaps   (real / synthetic / difference)
#
# Called from:
#   client_TVAE._train_generator()  — right after generator is frozen

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')        # safe for servers — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ───────────────────────────────────────────────
# KS TEST  (per feature)
# ───────────────────────────────────────────────
def ks_test_per_feature(real_df: pd.DataFrame,
                         syn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kolmogorov-Smirnov two-sample test per column.

    ks_statistic → 0   : distributions are identical
    ks_statistic → 1   : completely different
    p_value > 0.05     : cannot reject H0 (distributions are similar)
    """
    shared = [c for c in real_df.columns if c in syn_df.columns]
    rows   = []

    for col in shared:
        r = real_df[col].dropna().values.astype(float)
        s = syn_df[col].dropna().values.astype(float)
        if len(r) == 0 or len(s) == 0:
            continue
        stat, p = stats.ks_2samp(r, s)
        rows.append({
            'feature'     : col,
            'ks_statistic': round(float(stat), 4),
            'p_value'     : round(float(p),    4),
            'similar'     : bool(p > 0.05)
        })

    df = pd.DataFrame(rows).sort_values(
        'ks_statistic', ascending=False
    ).reset_index(drop=True)

    mean_ks   = float(df['ks_statistic'].mean()) if len(df) else 0.0
    n_similar = int(df['similar'].sum())

    print(f"     KS Test  ->  mean={mean_ks:.4f}  "
          f"similar={n_similar}/{len(df)}  "
          f"{'OK' if mean_ks < 0.05 else 'WARN' if mean_ks < 0.15 else 'POOR'}")
    return df


# ───────────────────────────────────────────────
# MANN-WHITNEY U TEST  (per feature)
# ───────────────────────────────────────────────
def mann_whitney_per_feature(real_df: pd.DataFrame,
                              syn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U two-sample test per column.

    Non-parametric — no normality assumption.
    Tests whether the distributions have the same median.

    p_value > 0.05 : cannot reject H0 (distributions are similar)
    """
    shared = [c for c in real_df.columns if c in syn_df.columns]
    rows   = []

    for col in shared:
        r = real_df[col].dropna().values.astype(float)
        s = syn_df[col].dropna().values.astype(float)
        if len(r) == 0 or len(s) == 0:
            continue
        try:
            stat, p = stats.mannwhitneyu(r, s, alternative='two-sided')
        except ValueError:
            stat, p = float('nan'), float('nan')
        rows.append({
            'feature'   : col,
            'mw_stat'   : round(float(stat), 4) if not np.isnan(stat) else None,
            'p_value'   : round(float(p),    4) if not np.isnan(p)    else None,
            'similar'   : bool(p > 0.05)        if not np.isnan(p)    else False
        })

    df = pd.DataFrame(rows).sort_values(
        'p_value', ascending=True
    ).reset_index(drop=True)

    p_vals    = df['p_value'].dropna().values
    n_similar = int((p_vals > 0.05).sum()) if len(p_vals) else 0
    mean_p    = float(p_vals.mean()) if len(p_vals) else 0.0

    print(f"     Mann-Whitney  ->  mean_p={mean_p:.4f}  "
          f"similar={n_similar}/{len(df)}  "
          f"{'OK' if mean_p > 0.05 else 'WARN' if mean_p > 0.01 else 'POOR'}")
    return df


# ───────────────────────────────────────────────
# WELCH'S T-TEST  (per feature)
# ───────────────────────────────────────────────
def ttest_per_feature(real_df: pd.DataFrame,
                       syn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Welch's independent two-sample t-test per column.

    Parametric — tests whether the means are equal.
    Does not assume equal variances (Welch variant).

    p_value > 0.05 : cannot reject H0 (means are similar)
    """
    shared = [c for c in real_df.columns if c in syn_df.columns]
    rows   = []

    for col in shared:
        r = real_df[col].dropna().values.astype(float)
        s = syn_df[col].dropna().values.astype(float)
        if len(r) < 2 or len(s) < 2:
            continue
        stat, p = stats.ttest_ind(r, s, equal_var=False)
        rows.append({
            'feature'  : col,
            't_stat'   : round(float(stat), 4),
            'p_value'  : round(float(p),    4),
            'similar'  : bool(p > 0.05)
        })

    df = pd.DataFrame(rows).sort_values(
        'p_value', ascending=True
    ).reset_index(drop=True)

    p_vals    = df['p_value'].dropna().values
    n_similar = int((p_vals > 0.05).sum()) if len(p_vals) else 0
    mean_p    = float(p_vals.mean()) if len(p_vals) else 0.0

    print(f"     T-Test  ->  mean_p={mean_p:.4f}  "
          f"similar={n_similar}/{len(df)}  "
          f"{'OK' if mean_p > 0.05 else 'WARN' if mean_p > 0.01 else 'POOR'}")
    return df


# ───────────────────────────────────────────────
# WASSERSTEIN DISTANCE  (per feature)
# ───────────────────────────────────────────────
def wasserstein_per_feature(real_df: pd.DataFrame,
                              syn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Earth Mover's Distance per column.
    0.0 = identical, larger = more different.
    """
    shared = [c for c in real_df.columns if c in syn_df.columns]
    rows   = []

    for col in shared:
        r = real_df[col].dropna().values.astype(float)
        s = syn_df[col].dropna().values.astype(float)
        if len(r) == 0 or len(s) == 0:
            continue
        dist = stats.wasserstein_distance(r, s)
        rows.append({
            'feature'            : col,
            'wasserstein_distance': round(float(dist), 4)
        })

    df = pd.DataFrame(rows).sort_values(
        'wasserstein_distance', ascending=False
    ).reset_index(drop=True)

    mean_wd = float(df['wasserstein_distance'].mean()) if len(df) else 0.0

    print(f"     Wasserstein ->  mean={mean_wd:.4f}  "
          f"{'OK' if mean_wd < 0.05 else 'WARN' if mean_wd < 0.10 else 'POOR'}")
    return df


# ───────────────────────────────────────────────
# TSTR  (Train on Synthetic, Test on Real)
# ───────────────────────────────────────────────
def tstr_score(real_X_np  : np.ndarray,
               syn_X_np   : np.ndarray,
               y_real     : np.ndarray,
               y_syn      : np.ndarray) -> dict:
    """
    Train a lightweight classifier on synthetic data, evaluate on real data.

    A good generator should produce synthetic data that is informative enough
    to train a classifier that generalises to the real distribution.

    Uses LogisticRegression (fast, no hyperparameters to tune).
    Also computes TRTS (Train Real, Test Synthetic) as an upper bound.

    Returns:
        {
          'tstr_acc'  : float  — accuracy trained on syn, tested on real
          'tstr_f1'   : float  — macro F1  trained on syn, tested on real
          'trts_acc'  : float  — accuracy trained on real, tested on syn
          'trts_f1'   : float  — macro F1  trained on real, tested on syn
        }
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score as sk_f1

    def _fit_eval(X_train, y_train, X_test, y_test):
        clf = LogisticRegression(
            max_iter    = 1000,
            solver      = 'lbfgs',
            multi_class = 'auto',
            C           = 1.0,
            random_state= 42,
        )
        try:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            f1  = float(sk_f1(y_test, preds, average='macro', zero_division=0))
        except Exception:
            acc, f1 = 0.0, 0.0
        return acc, f1

    tstr_acc, tstr_f1 = _fit_eval(syn_X_np,  y_syn,  real_X_np, y_real)
    trts_acc, trts_f1 = _fit_eval(real_X_np, y_real, syn_X_np,  y_syn)

    print(f"     TSTR  ->  acc={tstr_acc:.4f}  f1={tstr_f1:.4f}  "
          f"{'OK' if tstr_f1 > 0.60 else 'WARN' if tstr_f1 > 0.40 else 'POOR'}")
    print(f"     TRTS  ->  acc={trts_acc:.4f}  f1={trts_f1:.4f}  "
          f"(upper bound)")

    return {
        'tstr_acc' : round(tstr_acc, 4),
        'tstr_f1'  : round(tstr_f1,  4),
        'trts_acc' : round(trts_acc, 4),
        'trts_f1'  : round(trts_f1,  4),
    }


# ───────────────────────────────────────────────
# CORRELATION SIMILARITY  (pair level)
# ───────────────────────────────────────────────
def correlation_similarity(real_df: pd.DataFrame,
                            syn_df: pd.DataFrame) -> float:
    """
    Compares Pearson correlation matrices.
    Returns score in [0, 1].  1.0 = identical structure.
    """
    num_cols = [c for c in real_df.columns
                if c in syn_df.columns
                and pd.api.types.is_numeric_dtype(real_df[c])]

    if len(num_cols) < 2:
        return 1.0

    rc = real_df[num_cols].corr().values
    sc = syn_df[num_cols].corr().values

    mask     = np.triu(np.ones_like(rc, dtype=bool), k=1)
    rf, sf   = rc[mask], sc[mask]
    valid    = ~(np.isnan(rf) | np.isnan(sf))
    rf, sf   = rf[valid], sf[valid]

    if len(rf) == 0:
        return 1.0

    score = float(np.clip(1.0 - np.mean(np.abs(rf - sf)), 0, 1))
    return round(score, 4)


# ───────────────────────────────────────────────
# SDMETRICS QUALITY SCORE  (summary 0-1)
# ───────────────────────────────────────────────
def sdmetrics_quality_score(real_df: pd.DataFrame,
                              syn_df: pd.DataFrame) -> dict:
    """
    Single quality score combining column shape + pair similarity.

    column_shapes      = 1 - mean(KS statistic)
    column_pairs       = correlation matrix similarity
    overall            = mean of both
    mean_wasserstein   = mean Earth Mover's Distance across features
                         (lower = better; independent of PCA pre-processing)

    Reference (Tab-VAE paper, intrusion dataset): 93.9%
    Target for your IDS data: > 85%
    """
    ks_df           = ks_test_per_feature(real_df, syn_df)
    col_shape_score = float(1.0 - ks_df['ks_statistic'].mean()) \
                      if len(ks_df) else 0.0

    wd_df          = wasserstein_per_feature(real_df, syn_df)
    mean_wd        = float(wd_df['wasserstein_distance'].mean()) \
                     if len(wd_df) else 0.0

    col_pair_score = correlation_similarity(real_df, syn_df)
    overall        = (col_shape_score + col_pair_score) / 2.0

    print(f"     Quality  ->  "
          f"shapes={col_shape_score*100:.1f}%  "
          f"pairs={col_pair_score*100:.1f}%  "
          f"overall={overall*100:.1f}%  "
          f"mean_WD={mean_wd:.4f}  "
          f"{'EXCELLENT' if overall >= 0.90 else 'GOOD' if overall >= 0.75 else 'POOR'}")

    return {
        'column_shapes'     : round(col_shape_score, 4),
        'column_pair_trends': round(col_pair_score,  4),
        'overall'           : round(overall,          4),
        'mean_wasserstein'  : round(mean_wd,          4),
    }


# ───────────────────────────────────────────────
# PLOT: DISTRIBUTION OVERLAYS (KDE + histogram)
# ───────────────────────────────────────────────
def _plot_distributions(real_df: pd.DataFrame, syn_df: pd.DataFrame,
                         ks_df: pd.DataFrame,
                         domain_key: str, save_path: str,
                         n_cols_per_row: int = 7):
    """
    Plot histogram + KDE overlay for ALL features in a multi-row grid.

    Features are sorted by KS statistic (descending = worst first) so
    the top-left subplots immediately show where the generator struggles.

    Each subplot shows:
      - Blue histogram + KDE  : real data distribution
      - Orange histogram + KDE: synthetic data distribution
      - Title: feature name, KS statistic, p-value
    """
    cols = [c for c in real_df.columns if c in syn_df.columns]
    if not cols:
        return

    # sort all features: worst KS first (most informative order)
    ks_order = ks_df.set_index('feature')['ks_statistic'].to_dict()
    cols = sorted(cols, key=lambda c: ks_order.get(c, 0), reverse=True)

    n_total = len(cols)
    ncols   = min(n_cols_per_row, n_total)
    nrows   = (n_total + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 3.0 * nrows),
        constrained_layout=True
    )
    # flatten to 1-D regardless of shape
    axes_flat = np.array(axes).flatten() if n_total > 1 else [axes]

    ks_lookup = ks_df.set_index('feature') if len(ks_df) else pd.DataFrame()

    for ax, col in zip(axes_flat, cols):
        r = real_df[col].dropna().values.astype(float)
        s = syn_df[col].dropna().values.astype(float)

        ax.hist(r, bins=30, alpha=0.35, color='steelblue',
                density=True, label='Real')
        ax.hist(s, bins=30, alpha=0.35, color='darkorange',
                density=True, label='Synthetic')

        try:
            x_min = min(r.min(), s.min())
            x_max = max(r.max(), s.max())
            xs    = np.linspace(x_min, x_max, 200)
            ax.plot(xs, stats.gaussian_kde(r)(xs), color='steelblue',  lw=1.5)
            ax.plot(xs, stats.gaussian_kde(s)(xs), color='darkorange', lw=1.5)
        except Exception:
            pass

        if col in ks_lookup.index:
            ks_v = ks_lookup.loc[col, 'ks_statistic']
            p_v  = ks_lookup.loc[col, 'p_value']
            ax.set_title(f"{col}\nKS={ks_v:.3f}  p={p_v:.3f}", fontsize=7)
        else:
            ax.set_title(col, fontsize=7)

        ax.legend(fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_xlabel('value', fontsize=6)
        ax.set_ylabel('density', fontsize=6)

    # hide any unused subplot panels
    for ax in axes_flat[n_total:]:
        ax.set_visible(False)

    n_pass = int((ks_df['p_value'] > 0.05).sum()) if len(ks_df) else 0
    fig.suptitle(
        f'Real vs Synthetic — {domain_key}  '
        f'({n_pass}/{n_total} features pass KS p>0.05)\n'
        f'Sorted worst → best KS statistic (top-left = hardest)',
        fontsize=10
    )
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"     Plot saved -> {save_path}  [{nrows}×{ncols} grid, {n_total} features]")


# ───────────────────────────────────────────────
# PLOT: STATISTICAL SIGNIFICANCE SUMMARY
# ───────────────────────────────────────────────
def _plot_significance_summary(ks_df: pd.DataFrame,
                                mw_df: pd.DataFrame,
                                tt_df: pd.DataFrame,
                                domain_key: str,
                                save_path: str):
    """
    Three-panel bar chart showing p-values for every feature
    from KS, Mann-Whitney, and t-test.

    The horizontal red line at p=0.05 marks the significance
    threshold. Bars above the line → synthetic ≈ real.
    Features are sorted by KS p-value (ascending = hardest first).
    """
    # align on shared features sorted by KS p-value
    ks_sorted = ks_df.sort_values('p_value', ascending=True).reset_index(drop=True)
    feats     = ks_sorted['feature'].tolist()

    ks_p  = ks_sorted.set_index('feature')['p_value']
    mw_p  = mw_df.set_index('feature')['p_value'] if len(mw_df) else pd.Series(dtype=float)
    tt_p  = tt_df.set_index('feature')['p_value'] if len(tt_df) else pd.Series(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(feats) * 0.18), 9),
                              constrained_layout=True)

    panels = [
        (axes[0], ks_p,  'KS p-value',           'steelblue'),
        (axes[1], mw_p,  'Mann-Whitney p-value',  'seagreen'),
        (axes[2], tt_p,  "Welch's t-test p-value",'darkorange'),
    ]

    for ax, p_series, title, color in panels:
        vals = [float(p_series.get(f, 0.0)) for f in feats]
        ax.bar(range(len(feats)), vals, color=color, alpha=0.75, width=0.8)
        ax.axhline(0.05, color='red', linestyle='--', lw=1.2,
                   label='p=0.05 threshold')
        ax.set_xlim(-0.5, len(feats) - 0.5)
        ax.set_ylim(0, max(max(vals) * 1.1, 0.1))
        ax.set_ylabel('p-value', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xticks([])      # too many features to label
        ax.tick_params(labelsize=7)

    # x-label on bottom panel only
    n_above_ks = sum(1 for v in [float(ks_p.get(f, 0.0)) for f in feats]
                     if v > 0.05)
    axes[2].set_xlabel(
        f"Features sorted by KS p-value (ascending) — "
        f"{n_above_ks}/{len(feats)} features pass KS at p>0.05",
        fontsize=8
    )

    fig.suptitle(
        f'Statistical Significance Summary — {domain_key}',
        fontsize=11
    )
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"     Plot saved -> {save_path}")


# ───────────────────────────────────────────────
# PLOT: CORRELATION HEATMAPS
# ───────────────────────────────────────────────
def _plot_correlations(real_df: pd.DataFrame, syn_df: pd.DataFrame,
                        domain_key: str, save_path: str):
    num_cols = [c for c in real_df.columns
                if c in syn_df.columns
                and pd.api.types.is_numeric_dtype(real_df[c])][:20]
    if len(num_cols) < 2:
        return
    rc = real_df[num_cols].corr()
    sc = syn_df[num_cols].corr()
    dc = (rc - sc).abs()

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5),
                                      constrained_layout=True)
    kw = dict(cmap='coolwarm', vmin=-1, vmax=1,
              square=True, xticklabels=False, yticklabels=False)
    sns.heatmap(rc, ax=a1, **kw)
    a1.set_title('Real')
    sns.heatmap(sc, ax=a2, **kw)
    a2.set_title('Synthetic')
    sns.heatmap(dc, ax=a3, cmap='Reds', vmin=0, vmax=1,
                square=True, xticklabels=False, yticklabels=False)
    a3.set_title('|Difference| (lower = better)')
    fig.suptitle(f'Correlation Structure — {domain_key}', fontsize=11)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"     Plot saved -> {save_path}")


# ───────────────────────────────────────────────
# MAIN ENTRY POINT
# called from client_TVAE._train_generator()
#           and client_RVAE._train_generator()
# ───────────────────────────────────────────────
def evaluate_generator(domain_key  : str,
                        real_X_np   : np.ndarray,
                        generator   : dict,
                        n_synthetic : int  = 1000,
                        save_plots  : bool = True,
                        plots_dir   : str  = 'results/plots',
                        device      : str  = 'cpu',
                        syn_X_np    : np.ndarray = None,
                        y_real      : np.ndarray = None,
                        y_syn       : np.ndarray = None) -> dict:
    """
    Evaluate a frozen generator against the real data it learned.

    For TVAE: synthetic samples are generated in PCA space and then
    inverse-transformed to original feature space (comparison is
    apples-to-apples).

    For RVAE (and any generator that pre-generates data): pass
    syn_X_np directly. Synthesis is skipped and syn_X_np is used as-is.
    This decouples evaluate_generator from generator-specific APIs.

    Args:
        domain_key  : name of the domain  (used in filenames / titles)
        real_X_np   : real features (N, input_dim) — numpy float32,
                      already in original feature space
        generator   : client.generators[domain_key] dict.
                      For TVAE: must contain model, transformer,
                                cont_cols, cat_cols, pca.
                      For RVAE: only used for column name inference;
                                synthesis is done via syn_X_np.
        n_synthetic : how many synthetic samples to generate
                      (ignored when syn_X_np is provided)
        save_plots  : save PNG files (distributions + significance + correlation)
        plots_dir   : folder for plots
        device      : 'cpu' or 'cuda'
        syn_X_np    : optional pre-generated synthetic data
                      (N, input_dim) numpy float32.  When provided,
                      synthesis step is skipped entirely.
        y_real      : optional class labels for real_X_np (N,) int.
                      Required for TSTR. If None, TSTR is skipped.
        y_syn       : optional class labels for syn_X_np  (N,) int.
                      Required for TSTR. If None, TSTR is skipped.

    Returns:
        {
          'domain_key'  : str,
          'quality'     : { column_shapes, column_pair_trends, overall,
                            mean_wasserstein, tstr_f1, trts_f1 },
          'ks'          : pd.DataFrame,
          'mann_whitney': pd.DataFrame,
          'ttest'       : pd.DataFrame,
          'wasserstein' : pd.DataFrame,
          'tstr'        : dict or None,
        }
    """
    from tvae import synthesize

    print(f"\n  {'--'*23}")
    print(f"  [Eval] Generator: {domain_key}")
    print(f"  real={real_X_np.shape[0]}  synthetic={n_synthetic}")
    print(f"  {'--'*23}")

    # ── build real DataFrame (original feature space) ────
    n_features = real_X_np.shape[1]
    col_names  = [f'f_{i}' for i in range(n_features)]
    real_df    = pd.DataFrame(real_X_np.astype(np.float32),
                               columns=col_names)

    # ── get synthetic data ────────────────────────────────
    # Path A: syn_X_np provided (RVAE, or any pre-generated data)
    #         → use directly, already in original feature space
    # Path B: no syn_X_np (TVAE) → synthesise in PCA space,
    #         then inverse-transform to original feature space
    if syn_X_np is not None:
        X_syn  = np.clip(syn_X_np, 0.0, 1.0).astype(np.float32)
        syn_df = pd.DataFrame(X_syn, columns=col_names)
        print(f"  Using pre-generated synthetic data: {X_syn.shape}")
    else:
        from tvae import synthesize
        syn_pca_df = synthesize(
            model            = generator['model'],
            transformer      = generator['transformer'],
            n                = n_synthetic,
            continuous_cols  = generator['cont_cols'],
            categorical_cols = generator['cat_cols'],
            device           = device
        )
        pca = generator.get('pca')
        if pca is not None:
            X_pca_syn = syn_pca_df.values.astype(np.float32)
            X_syn     = pca.inverse_transform(X_pca_syn)
            X_syn     = np.clip(X_syn, 0.0, 1.0).astype(np.float32)
            syn_df    = pd.DataFrame(X_syn, columns=col_names)
            print(f"  Inverse-PCA applied: {X_pca_syn.shape[1]}d → {X_syn.shape[1]}d")
        else:
            pca_cols  = [f'pc_{i}' for i in range(syn_pca_df.shape[1])]
            syn_df    = syn_pca_df.copy()
            syn_df.columns = pca_cols
            real_df   = real_df[real_df.columns[:syn_pca_df.shape[1]]].copy()
            real_df.columns = pca_cols
            col_names = pca_cols
            print("  Warning: no PCA stored — comparing in PCA space")

    # ── statistical tests ────────────────────────────────
    print(f"\n  Statistical tests ({len(col_names)} features):")
    ks_df = ks_test_per_feature(real_df, syn_df)
    mw_df = mann_whitney_per_feature(real_df, syn_df)
    tt_df = ttest_per_feature(real_df, syn_df)
    wd_df = wasserstein_per_feature(real_df, syn_df)

    # ── quality score (includes mean Wasserstein) ─────────
    quality = sdmetrics_quality_score(real_df, syn_df)

    # ── TSTR / TRTS ───────────────────────────────────────
    tstr = None
    if y_real is not None and y_syn is not None:
        print(f"\n  TSTR / TRTS:")
        tstr = tstr_score(
            real_X_np = real_X_np,
            syn_X_np  = X_syn,
            y_real    = y_real,
            y_syn     = y_syn,
        )
        quality['tstr_f1'] = tstr['tstr_f1']
        quality['trts_f1'] = tstr['trts_f1']
    else:
        quality['tstr_f1'] = None
        quality['trts_f1'] = None

    # ── significance summary stats ───────────────────────
    ks_pass = int((ks_df['p_value'] > 0.05).sum())
    mw_pass = int((mw_df['p_value'].dropna() > 0.05).sum()) if len(mw_df) else 0
    tt_pass = int((tt_df['p_value'].dropna() > 0.05).sum()) if len(tt_df) else 0
    n_feat  = len(ks_df)

    print(f"\n  Significance summary (p > 0.05 = distributions match):")
    print(f"    KS test        : {ks_pass}/{n_feat} features pass")
    print(f"    Mann-Whitney U : {mw_pass}/{n_feat} features pass")
    print(f"    Welch's t-test : {tt_pass}/{n_feat} features pass")

    # ── plots ─────────────────────────────────────────────
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        safe = domain_key.replace('/', '_').replace(' ', '_')

        # distribution overlay — ALL features in a multi-row grid
        # 7 subplots per row; sorted worst→best KS statistic
        _plot_distributions(
            real_df, syn_df,
            ks_df          = ks_df,
            domain_key     = domain_key,
            n_cols_per_row = 7,
            save_path      = os.path.join(plots_dir,
                                          f'{safe}_distributions.png')
        )

        # statistical significance summary
        _plot_significance_summary(
            ks_df, mw_df, tt_df,
            domain_key = domain_key,
            save_path  = os.path.join(plots_dir,
                                      f'{safe}_significance.png')
        )

        # correlation heatmaps (up to 20 features)
        _plot_correlations(
            real_df, syn_df, domain_key,
            save_path = os.path.join(plots_dir,
                                     f'{safe}_correlations.png')
        )

    return {
        'domain_key'  : domain_key,
        'quality'     : quality,
        'ks'          : ks_df,
        'mann_whitney': mw_df,
        'ttest'       : tt_df,
        'wasserstein' : wd_df,
        'tstr'        : tstr,
    }
