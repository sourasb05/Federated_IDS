"""
GMM-based Generative Replay Pipeline for Continual Learning
============================================================
Mirrors tabddpm_generative_replay_pytorch.py structure exactly.

Uses per-class Bayesian Gaussian Mixture Models (BayesianGMM) instead of a
neural generative model. GMM is the lightest viable generative replay baseline:

  - No gradient training — closed-form EM, converges in seconds
  - No KL collapse, no posterior collapse, no hyperparameter instability
  - BayesianGMM automatically prunes unused components (Dirichlet process prior)
  - Per-class fitting → labels are always exact at sample time
  - 'diag' covariance → scales to high-dimensional flat windows without singularity

Key differences from TVAE / TabDDPM:
  - No neural network, no GPU needed
  - 'diag' covariance: assumes feature independence within a component
    (full covariance is numerically unstable for T*F dimensional inputs)
  - Standardisation before fit, de-standardisation after sample
    (same as TabDDPM — GMM also assumes roughly unit-scale data)

Flow:
  t0:  Fit per-class GMMs on D0
  t1:  Sample D0_hat from frozen GMMs
       Train on D1 ∪ D0_hat, refit GMMs on combined
  t2:  Repeat

Reference:
  Reynolds (2009) Gaussian Mixture Models. Encyclopedia of Biometrics.
  Blei & Jordan (2006) Variational inference for Dirichlet process mixtures.
"""

import os
import math
import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import ks_2samp, wasserstein_distance, gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. CORRELATION STRUCTURE VISUALISATION  (same as TabDDPM file)
# ============================================================

def plot_correlation_comparison(real_df, synthetic_df, feature_columns,
                                title="Correlation Structure", save_path=None,
                                figsize=(18, 5)):
    real_corr  = real_df[feature_columns].corr()
    synth_corr = synthetic_df[feature_columns].corr()
    diff_corr  = (real_corr - synth_corr).abs()

    labels = [c[:8] for c in feature_columns]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    for ax, mat, ttl, cmap, vmin, vmax in [
        (axes[0], real_corr,  'Real',                      'RdBu_r', -1, 1),
        (axes[1], synth_corr, 'Synthetic',                 'RdBu_r', -1, 1),
        (axes[2], diff_corr,  'Difference (lower=better)', 'Reds',    0, 1),
    ]:
        im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(ttl, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")
    plt.show()
    plt.close()

    mean_diff = diff_corr.values[np.triu_indices_from(diff_corr.values, k=1)].mean()
    max_diff  = diff_corr.values[np.triu_indices_from(diff_corr.values, k=1)].max()
    print(f"  Mean |corr difference|: {mean_diff:.4f}")
    print(f"  Max  |corr difference|: {max_diff:.4f}")
    return real_corr, synth_corr, diff_corr


def plot_correlation_comparison_per_class(real_df, synthetic_df, feature_columns,
                                          label_column='label',
                                          save_path_prefix=None, figsize=(18, 5)):
    for cls in sorted(real_df[label_column].unique()):
        real_cls  = real_df[real_df[label_column] == cls]
        synth_cls = synthetic_df[synthetic_df[label_column] == cls]
        if len(synth_cls) < 2:
            print(f"  Skipping class {cls}: too few synthetic samples")
            continue
        save_path = f"{save_path_prefix}_class_{cls}.png" if save_path_prefix else None
        print(f"\n{'='*60}")
        print(f"Class {cls} — Real: {len(real_cls)}, Synthetic: {len(synth_cls)}")
        print(f"{'='*60}")
        plot_correlation_comparison(
            real_cls, synth_cls, feature_columns,
            title=f"Correlation Structure — Class {cls} (GMM)",
            save_path=save_path, figsize=figsize
        )


# ============================================================
# 2. PER-CLASS GMM MODEL
# ============================================================

class PerClassGMM:
    """
    BayesianGaussianMixture wrapper for one class.

    Why 'diag' covariance:
      Full covariance requires estimating D*(D+1)/2 parameters per component.
      For D=14 features that is 105 params/component — manageable.
      For flat windows D=T*F=140 that is 9870 params/component — severely
      underdetermined with typical sample sizes.
      'diag' reduces this to D params/component regardless of D.

    Standardisation:
      GMM assumes roughly unit-scale data. We z-score each feature
      (mean=0, std=1) before fitting and reverse the transform at sample time.
      This prevents high-magnitude features from dominating the mixture.
    """

    def __init__(self,
                 n_components : int   = 10,
                 covariance_type: str = 'diag',
                 max_iter      : int  = 300,
                 n_init        : int  = 3,
                 random_state  : int  = 42):

        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.max_iter        = max_iter
        self.n_init          = n_init
        self.random_state    = random_state

        self._gmm  = None
        self._mean : np.ndarray = None
        self._std  : np.ndarray = None

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / (self._std + 1e-8)

    def _destandardise(self, X: np.ndarray) -> np.ndarray:
        return X * (self._std + 1e-8) + self._mean

    def fit(self, X_np: np.ndarray) -> 'PerClassGMM':
        """Fit on (N, D) float32 feature matrix."""
        self._mean = X_np.mean(axis=0).astype(np.float32)
        self._std  = X_np.std(axis=0).astype(np.float32)
        X_std = self._standardise(X_np)

        # cap components to avoid fitting errors on small classes
        n_comp = min(self.n_components, max(1, len(X_np) // 5))

        self._gmm = BayesianGaussianMixture(
            n_components                    = n_comp,
            covariance_type                 = self.covariance_type,
            weight_concentration_prior_type = 'dirichlet_process',
            weight_concentration_prior      = 1e-2,   # sparse — prunes unused modes
            max_iter                        = self.max_iter,
            n_init                          = self.n_init,
            random_state                    = self.random_state,
            reg_covar                       = 1e-4,   # numerical stability
        )
        self._gmm.fit(X_std)

        effective = int((self._gmm.weights_ > 1e-3).sum())
        print(f"    GMM fitted: {effective}/{n_comp} effective components")
        return self

    def generate(self, n: int) -> np.ndarray:
        """Sample n rows. Returns (n, D) float32 in original feature space."""
        X_std, _ = self._gmm.sample(n)
        return self._destandardise(X_std.astype(np.float32))

    @property
    def effective_components(self) -> int:
        if self._gmm is None:
            return 0
        return int((self._gmm.weights_ > 1e-3).sum())


# ============================================================
# 3. GENERATIVE REPLAY PIPELINE  (Per-class GMMs)
# ============================================================

class GMMGenerativeReplayPipeline:
    """
    Continual learning with GMM generative replay.

    Mirrors TabDDPMGenerativeReplayPipeline from tabddpm_generative_replay_pytorch.py.

    KEY DESIGN: One BayesianGMM per class.
      - Each class has its own GMM fitted on that class's features only
      - At replay: sample proportionally from each class GMM
      - Refit all GMMs on combined (real + replay) data each timestep
      - Labels are exact — GMM_c.generate() always returns class-c samples
    """

    def __init__(self,
                 feature_columns  : list,
                 label_column     : str   = 'label',
                 n_components     : int   = 10,
                 covariance_type  : str   = 'diag',
                 max_iter         : int   = 300,
                 n_init           : int   = 3,
                 replay_ratio     : float = 1.0):

        self.feature_columns = feature_columns
        self.label_column    = label_column
        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.max_iter        = max_iter
        self.n_init          = n_init
        self.replay_ratio    = replay_ratio

        self.class_gmms  : dict = {}    # label -> PerClassGMM
        self.classifier          = None
        self.classes             = None
        self.history     : list  = []
        self.timestep    : int   = 0

    # ── internal helpers ─────────────────────────────────────

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.feature_columns + [self.label_column]].copy()

    def _train_class_gmms(self, df: pd.DataFrame, verbose: bool = True):
        """Fit one PerClassGMM per class (features only, no label column)."""
        self.classes = sorted(df[self.label_column].unique())

        for cls_label in self.classes:
            cls_data = df[df[self.label_column] == cls_label][self.feature_columns]
            X_np = cls_data.values.astype(np.float32)

            if verbose:
                print(f"\n  -- Class {cls_label} ({len(X_np)} samples) --")

            gmm = PerClassGMM(
                n_components    = self.n_components,
                covariance_type = self.covariance_type,
                max_iter        = self.max_iter,
                n_init          = self.n_init,
            )
            gmm.fit(X_np)
            self.class_gmms[cls_label] = gmm

    def _generate_replay(self, n_per_class: int) -> pd.DataFrame:
        """Generate balanced replay samples from per-class GMMs."""
        all_dfs = []
        for cls_label in self.classes:
            X_syn  = self.class_gmms[cls_label].generate(n_per_class)
            df_syn = pd.DataFrame(X_syn, columns=self.feature_columns)
            df_syn[self.label_column] = cls_label
            all_dfs.append(df_syn)
        return pd.concat(all_dfs, ignore_index=True)

    def _quality_check(self, real_data: pd.DataFrame):
        """Compare real vs synthetic feature statistics per class."""
        print("  [Quality Check]")
        for cls_label in self.classes:
            real_cls = real_data[real_data[self.label_column] == cls_label]
            X_syn    = self.class_gmms[cls_label].generate(len(real_cls))
            syn_df   = pd.DataFrame(X_syn, columns=self.feature_columns)

            print(f"\n  Class {cls_label}:")
            print(f"  {'Column':<14} {'Real Mean':>12} {'Synth Mean':>12} "
                  f"{'Real Std':>12} {'Synth Std':>12}")
            print(f"  {'-'*62}")
            for col in self.feature_columns[:5]:
                rm = real_cls[col].mean()
                rs = real_cls[col].std()
                sm = syn_df[col].mean()
                ss = syn_df[col].std()
                print(f"  {col:<14} {rm:>12.4f} {sm:>12.4f} {rs:>12.4f} {ss:>12.4f}")

    def _evaluate(self, train_data: pd.DataFrame,
                  test_df: pd.DataFrame = None) -> dict:
        """
        TRTS evaluation: Train on Real, Test on Synthetic.
        Also tests on held-out real data if provided.
        """
        X_train = train_data[self.feature_columns].values
        y_train = train_data[self.label_column].values

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        self.classifier = clf

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        print(f"\n  [Train on Real] Train accuracy: {train_acc:.4f}")

        result = {
            'timestep'         : self.timestep,
            'n_real'           : len(train_data),
            'n_synthetic'      : 0,
            'train_on_real_acc': train_acc,
        }

        # TRTS
        n_per_class  = len(train_data) // max(len(self.classes), 1)
        synthetic_df = self.generate_synthetic(n_per_class)
        X_synth      = synthetic_df[self.feature_columns].values
        y_synth      = synthetic_df[self.label_column].values
        trts_acc     = accuracy_score(y_synth, clf.predict(X_synth))
        result['trts_acc'] = trts_acc
        print(f"  [Test on Synthetic] TRTS accuracy: {trts_acc:.4f}")
        print(f"\n  TRTS Classification Report:\n"
              f"{classification_report(y_synth, clf.predict(X_synth))}")

        if test_df is not None:
            td       = self._get_features(test_df)
            X_test   = td[self.feature_columns].values
            y_test   = td[self.label_column].values
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            result['test_real_acc'] = test_acc
            print(f"  [Test on Real holdout] accuracy: {test_acc:.4f}")

        return result

    # ── public API ───────────────────────────────────────────

    def initial_train(self, df_train: pd.DataFrame,
                      df_test: pd.DataFrame = None) -> 'GMMGenerativeReplayPipeline':
        """t0: Initial training."""
        print("=" * 60)
        print(f"TIME STEP t{self.timestep}: Initial Training")
        print("=" * 60)

        data = self._get_features(df_train)
        print(f"Samples: {len(data)} | "
              f"Label dist: {dict(data[self.label_column].value_counts())}")

        print("\n[1/2] Fitting per-class GMMs...")
        self._train_class_gmms(data)

        print("\n[2/2] Evaluating (TRTS)...")
        result = self._evaluate(data, df_test)
        self._quality_check(data)
        self.history.append(result)
        self.timestep += 1
        return self

    def continual_update(self, df_new: pd.DataFrame,
                         df_test: pd.DataFrame = None) -> 'GMMGenerativeReplayPipeline':
        """t_k (k>0): Generate replay, combine, refit GMMs."""
        print("\n" + "=" * 60)
        print(f"TIME STEP t{self.timestep}: Continual Update")
        print("=" * 60)

        new_data    = self._get_features(df_new)
        n_new       = len(new_data)
        n_per_class = int(n_new * self.replay_ratio / max(len(self.classes), 1))

        print(f"\n[1/3] Generating replay: {n_per_class} samples × "
              f"{len(self.classes)} classes = {n_per_class * len(self.classes)} total")
        replay_df = self._generate_replay(n_per_class)
        print(f"  Replay label dist: {dict(replay_df[self.label_column].value_counts())}")

        combined = pd.concat([new_data, replay_df], ignore_index=True)
        print(f"\n[2/3] Combined: {len(new_data)} real + {len(replay_df)} synthetic "
              f"= {len(combined)} total")
        print(f"  Combined label dist: {dict(combined[self.label_column].value_counts())}")

        print(f"\n[3/3] Refitting per-class GMMs on combined data...")
        self._train_class_gmms(combined)

        print("\n  Evaluating (TRTS)...")
        result = self._evaluate(combined, df_test)
        result['n_real']      = n_new
        result['n_synthetic'] = len(replay_df)
        self._quality_check(combined)
        self.history.append(result)
        self.timestep += 1
        return self

    def generate_synthetic(self, n_per_class: int = 100) -> pd.DataFrame:
        """Generate a full synthetic DataFrame with labels."""
        all_dfs = []
        for cls_label in self.classes:
            X_syn  = self.class_gmms[cls_label].generate(n_per_class)
            df_syn = pd.DataFrame(X_syn, columns=self.feature_columns)
            df_syn[self.label_column] = cls_label
            all_dfs.append(df_syn)
        return pd.concat(all_dfs, ignore_index=True)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("GMM GENERATIVE REPLAY SUMMARY")
        print("=" * 60)
        print(f"  {'Step':<6} {'Real':<7} {'Synth':<7} "
              f"{'Train(R)':<10} {'TRTS':<10} {'Test(R)':<10}")
        print(f"  {'-'*56}")
        for h in self.history:
            train_r = f"{h.get('train_on_real_acc', 0):.4f}"
            trts    = f"{h.get('trts_acc', 0):.4f}" if 'trts_acc' in h else 'N/A'
            test_r  = f"{h.get('test_real_acc', 0):.4f}" if 'test_real_acc' in h else 'N/A'
            print(f"  t{h['timestep']:<5} {h['n_real']:<7} {h['n_synthetic']:<7} "
                  f"{train_r:<10} {trts:<10} {test_r:<10}")
        print()
        print("  Train(R) = classifier trained & tested on real data")
        print("  TRTS     = trained on real, tested on synthetic (higher = better generation)")
        print("  Test(R)  = trained on real, tested on held-out real data")

    def plot_correlation(self, real_df: pd.DataFrame, n_synthetic: int = None,
                         per_class: bool = True, save_path_prefix: str = None):
        real_data    = self._get_features(real_df)
        n_per_class  = (len(real_data) // len(self.classes)
                        if n_synthetic is None else n_synthetic // len(self.classes))
        synthetic_df = self.generate_synthetic(n_per_class)

        save_overall = f"{save_path_prefix}_overall.png" if save_path_prefix else None
        print("\n" + "=" * 60)
        print("OVERALL Correlation Comparison")
        print("=" * 60)
        plot_correlation_comparison(
            real_data, synthetic_df, self.feature_columns,
            title="Correlation Structure — Overall (GMM)",
            save_path=save_overall
        )

        if per_class:
            plot_correlation_comparison_per_class(
                real_data, synthetic_df, self.feature_columns,
                label_column=self.label_column,
                save_path_prefix=save_path_prefix
            )

    def evaluate_synthetic_quality(self,
                                   real_df       : pd.DataFrame,
                                   n_synthetic   : Optional[int] = None,
                                   save_dir      : str = 'results',
                                   plots_per_row : int = 5) -> pd.DataFrame:
        """
        Compare real vs synthetic distributions with:
          1. KS test (per feature, per class)         — distributional similarity
          2. Wasserstein distance (per feature)        — earth-mover's distance
          3. Distribution overlay plots (KDE + hist)   — visual per-feature comparison
          4. KS summary bar chart                      — p-values across all features

        Parameters
        ----------
        real_df      : full real DataFrame (must contain label column)
        n_synthetic  : samples to generate per class (default: match real per-class count)
        save_dir     : directory for saved PNGs and CSV
        plots_per_row: feature subplots per row in distribution grid

        Returns
        -------
        ks_summary : pd.DataFrame  [class, feature, ks_stat, p_value, wasserstein, similar]
                     sorted by ks_stat descending (worst first)
        """
        os.makedirs(save_dir, exist_ok=True)
        real_data = self._get_features(real_df)

        n_classes = len(self.classes) if self.classes else 1
        n_per_class = (len(real_data) // max(n_classes, 1)
                       if n_synthetic is None
                       else n_synthetic // max(n_classes, 1))

        all_rows = []

        for cls_label in self.classes:
            real_cls = real_data[real_data[self.label_column] == cls_label][self.feature_columns]
            X_syn    = self.class_gmms[cls_label].generate(n_per_class)
            syn_cls  = pd.DataFrame(X_syn, columns=self.feature_columns)

            print(f"\n{'='*60}")
            print(f"  Class {cls_label} | real={len(real_cls)}  synthetic={len(syn_cls)}")
            print(f"{'='*60}")
            print(f"  {'Feature':<16} {'KS stat':>9} {'p-value':>9} "
                  f"{'Wasser.':>9} {'Similar?':>10}")
            print(f"  {'-'*56}")

            feat_rows = []
            for col in self.feature_columns:
                r = real_cls[col].dropna().values.astype(float)
                s = syn_cls[col].dropna().values.astype(float)
                if len(r) < 2 or len(s) < 2:
                    continue
                ks_stat, p_val = ks_2samp(r, s)
                wd             = wasserstein_distance(r, s)
                similar        = p_val > 0.05
                feat_rows.append({
                    'class'      : cls_label,
                    'feature'    : col,
                    'ks_stat'    : round(ks_stat, 4),
                    'p_value'    : round(p_val,   4),
                    'wasserstein': round(wd,       4),
                    'similar'    : similar,
                })
                flag = 'OK  ' if similar else 'FAIL'
                print(f"  {col:<16} {ks_stat:>9.4f} {p_val:>9.4f} "
                      f"{wd:>9.4f} {flag:>10}")

            all_rows.extend(feat_rows)

            # ── summary stats ───────────────────────────────────
            if feat_rows:
                mean_ks   = np.mean([r['ks_stat']     for r in feat_rows])
                n_similar = sum(1 for r in feat_rows if r['similar'])
                n_feat    = len(feat_rows)
                rating    = ('EXCELLENT' if mean_ks < 0.05 else
                             'GOOD'      if mean_ks < 0.15 else 'POOR')
                print(f"\n  Mean KS = {mean_ks:.4f} | "
                      f"{n_similar}/{n_feat} features similar (p>0.05) | {rating}")

            # ── distribution overlay plot ────────────────────────
            feat_rows_sorted = sorted(feat_rows,
                                      key=lambda r: r['ks_stat'], reverse=True)
            n_feat  = len(feat_rows_sorted)
            ncols   = min(plots_per_row, n_feat)
            nrows   = math.ceil(n_feat / ncols) if ncols > 0 else 1

            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(4.0 * ncols, 3.2 * nrows),
                                     constrained_layout=True)
            axes_flat = np.array(axes).flatten() if n_feat > 1 else [axes]

            for ax, row in zip(axes_flat, feat_rows_sorted):
                col = row['feature']
                r   = real_cls[col].dropna().values.astype(float)
                s   = syn_cls[col].dropna().values.astype(float)

                ax.hist(r, bins=30, alpha=0.35, color='steelblue',
                        density=True, label='Real')
                ax.hist(s, bins=30, alpha=0.35, color='darkorange',
                        density=True, label='Synthetic')

                try:
                    xs = np.linspace(min(r.min(), s.min()),
                                     max(r.max(), s.max()), 300)
                    ax.plot(xs, gaussian_kde(r)(xs), color='steelblue',  lw=1.5)
                    ax.plot(xs, gaussian_kde(s)(xs), color='darkorange', lw=1.5)
                except Exception:
                    pass

                status = 'OK' if row['similar'] else 'FAIL'
                ax.set_title(
                    f"{col}\nKS={row['ks_stat']:.3f}  "
                    f"p={row['p_value']:.3f}  [{status}]",
                    fontsize=7
                )
                ax.legend(fontsize=6)
                ax.tick_params(labelsize=5)
                ax.set_xlabel('value',   fontsize=6)
                ax.set_ylabel('density', fontsize=6)

            for ax in axes_flat[n_feat:]:
                ax.set_visible(False)

            n_sim = sum(1 for r in feat_rows if r['similar'])
            fig.suptitle(
                f"Real vs Synthetic — Class {cls_label}  "
                f"({n_sim}/{n_feat} features pass KS p>0.05)\n"
                f"GMM | sorted worst → best KS (top-left = hardest)",
                fontsize=10
            )
            dist_path = os.path.join(save_dir, f'gmm_dist_class{cls_label}.png')
            plt.savefig(dist_path, dpi=110, bbox_inches='tight')
            plt.close(fig)
            print(f"  Distribution plot saved → {dist_path}")

            # ── KS p-value + statistic bar charts ───────────────
            feat_rows_asc = sorted(feat_rows, key=lambda r: r['p_value'])
            feat_names    = [r['feature'] for r in feat_rows_asc]
            p_values      = [r['p_value'] for r in feat_rows_asc]
            ks_stats      = [r['ks_stat'] for r in feat_rows_asc]

            fig2, (ax1, ax2) = plt.subplots(2, 1,
                                             figsize=(max(10, n_feat * 0.35), 8),
                                             constrained_layout=True)

            colors = ['#d73027' if p <= 0.05 else '#4575b4' for p in p_values]
            ax1.bar(range(n_feat), p_values, color=colors, alpha=0.85, width=0.7)
            ax1.axhline(0.05, color='black', linestyle='--', lw=1.5,
                        label='p = 0.05 threshold')
            ax1.set_ylabel('KS p-value', fontsize=9)
            ax1.set_title(
                f'KS Test p-values — Class {cls_label}  '
                f'(blue = similar, red = different)',
                fontsize=10
            )
            ax1.set_xticks(range(n_feat))
            ax1.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=7)
            ax1.legend(fontsize=8)
            ax1.set_ylim(0, max(max(p_values) * 1.15, 0.1))

            ax2.bar(range(n_feat), ks_stats, color='#fc8d59', alpha=0.85, width=0.7)
            ax2.axhline(0.05, color='black', linestyle='--', lw=1.5,
                        label='KS = 0.05 threshold')
            ax2.set_ylabel('KS statistic (lower = better)', fontsize=9)
            ax2.set_title(
                f'KS Statistic — Class {cls_label}  '
                f'(lower = real ≈ synthetic)',
                fontsize=10
            )
            ax2.set_xticks(range(n_feat))
            ax2.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=7)
            ax2.legend(fontsize=8)

            fig2.suptitle(
                f'KS Test Summary — GMM — Class {cls_label}',
                fontsize=12, fontweight='bold'
            )
            ks_path = os.path.join(save_dir, f'gmm_ks_class{cls_label}.png')
            plt.savefig(ks_path, dpi=110, bbox_inches='tight')
            plt.close(fig2)
            print(f"  KS summary plot saved   → {ks_path}")

        # ── combined summary table ───────────────────────────────
        ks_summary = (pd.DataFrame(all_rows)
                        .sort_values('ks_stat', ascending=False)
                        .reset_index(drop=True))

        print(f"\n{'='*60}")
        print(f"  OVERALL KS SUMMARY  (all classes, all features)")
        print(f"{'='*60}")
        mean_ks_all = ks_summary['ks_stat'].mean()
        n_sim_all   = ks_summary['similar'].sum()
        n_total     = len(ks_summary)
        print(f"  Mean KS statistic : {mean_ks_all:.4f}")
        print(f"  Features similar  : {n_sim_all}/{n_total} (p > 0.05)")
        rating = ('EXCELLENT' if mean_ks_all < 0.05 else
                  'GOOD'      if mean_ks_all < 0.15 else 'POOR')
        print(f"  Overall rating    : {rating}")

        csv_path = os.path.join(save_dir, 'gmm_ks_results.csv')
        ks_summary.to_csv(csv_path, index=False)
        print(f"  KS results CSV    → {csv_path}")

        return ks_summary

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'gmm_pipeline_state.pkl'), 'wb') as f:
            pickle.dump({
                'class_gmms'     : self.class_gmms,
                'classifier'     : self.classifier,
                'classes'        : self.classes,
                'feature_columns': self.feature_columns,
                'label_column'   : self.label_column,
                'history'        : self.history,
                'timestep'       : self.timestep,
            }, f)
        print(f"  Pipeline saved to {directory}/")

    @classmethod
    def load(cls, directory: str) -> 'GMMGenerativeReplayPipeline':
        with open(os.path.join(directory, 'gmm_pipeline_state.pkl'), 'rb') as f:
            state = pickle.load(f)
        obj = cls(feature_columns=state['feature_columns'],
                  label_column=state['label_column'])
        obj.class_gmms   = state['class_gmms']
        obj.classifier   = state['classifier']
        obj.classes      = state['classes']
        obj.history      = state['history']
        obj.timestep     = state['timestep']
        return obj


# ============================================================
# 4. DEMO
# ============================================================

if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)

    # ── Load data ─────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(
        "/Users/souba636/Documents/Research/Projects/vinnova_paper_2/"
        "Federated_IDS/attack_data/worstparent_var15_base/"
        "1_features_timeseries_60_sec.csv"
    )

    drop_cols = ['Unnamed: 0', 'disr', 'diss', 'disr.1', 'diss.1']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    feature_cols = [c for c in df.columns if c != 'label']
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Total samples: {len(df)}\n")
    print(f"Label distribution (full): {dict(df['label'].value_counts())}")

    # ── Stratified shuffle before splitting ───────────────────
    # The raw CSV is temporally ordered: all benign rows come first,
    # then all attack rows.  Without shuffling, chunk t0 contains ONLY
    # class 0 — the GMM never sees class 1, so it cannot model attack
    # traffic and every KS test fails on class 1 features.
    # Stratified shuffle preserves the overall class ratio in every chunk.
    from sklearn.model_selection import train_test_split

    df_class0 = df[df['label'] == 0.0].sample(frac=1, random_state=42).reset_index(drop=True)
    df_class1 = df[df['label'] == 1.0].sample(frac=1, random_state=42).reset_index(drop=True)

    def stratified_chunks(df0, df1, n_chunks=3):
        """Split each class into n_chunks equal parts, then zip them."""
        chunks = []
        s0 = len(df0) // n_chunks
        s1 = len(df1) // n_chunks
        for i in range(n_chunks):
            part0 = df0.iloc[i*s0:(i+1)*s0]
            part1 = df1.iloc[i*s1:(i+1)*s1]
            chunks.append(pd.concat([part0, part1]).sample(frac=1, random_state=42+i).reset_index(drop=True))
        return chunks

    chunks = stratified_chunks(df_class0, df_class1, n_chunks=3)
    df_t0, df_t1, df_t2 = chunks

    # Hold out 100 samples from the shuffled full data for global evaluation
    global_test = df.sample(n=100, random_state=99)
    print(f"\nStratified split:")
    for i, c in enumerate(chunks):
        print(f"  t{i}: {len(c)} rows | labels: {dict(c['label'].value_counts())}")
    print(f"  global_test: {len(global_test)} rows\n")

    # ── Run GMM pipeline ──────────────────────────────────────
    pipeline = GMMGenerativeReplayPipeline(
        feature_columns = feature_cols,
        label_column    = 'label',
        n_components    = 10,          # max components — Bayesian prior prunes unused
        covariance_type = 'diag',      # diagonal covariance — stable for many features
        max_iter        = 300,
        n_init          = 3,
        replay_ratio    = 1.0,
    )

    pipeline.initial_train(df_t0, df_test=global_test)
    pipeline.continual_update(df_t1, df_test=global_test)
    pipeline.continual_update(df_t2, df_test=global_test)

    # ── Baseline: naive (no replay) ───────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE: Naive training (no replay, latest chunk only)")
    print("=" * 60)
    for i, chunk_df in enumerate([df_t0, df_t1, df_t2]):
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(chunk_df[feature_cols].values, chunk_df['label'].values)
        acc = accuracy_score(
            global_test['label'].values,
            clf.predict(global_test[feature_cols].values)
        )
        print(f"  t{i} only -> Global test acc: {acc:.4f}")

    # ── Skyline: all data at once ──────────────────────────────
    print("\n" + "=" * 60)
    print("SKYLINE: Train on ALL data (no continual learning)")
    print("=" * 60)
    clf_all = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_all.fit(df[feature_cols].values, df['label'].values)
    acc_all = accuracy_score(
        global_test['label'].values,
        clf_all.predict(global_test[feature_cols].values)
    )
    print(f"  All data -> Global test acc: {acc_all:.4f}")

    # ── Summary ───────────────────────────────────────────────
    pipeline.print_summary()

    # ── Distribution comparison + KS tests ───────────────────
    # Compare synthetic samples against the COMBINED data that the GMM
    # was actually trained on (t0 + t1 + t2 combined after all updates).
    # Evaluating against the raw df (temporally ordered) would be unfair:
    # the full csv has attack features ranging up to 185+ while t0 has
    # values up to ~2 — a guaranteed KS failure unrelated to GMM quality.
    df_combined = pd.concat([df_t0, df_t1, df_t2], ignore_index=True)
    print("\n" + "=" * 60)
    print("DISTRIBUTION COMPARISON & KS STATISTICAL TESTS")
    print(f"  Evaluating against combined training data ({len(df_combined)} rows)")
    print("=" * 60)
    ks_results = pipeline.evaluate_synthetic_quality(
        real_df       = df_combined,
        n_synthetic   = len(df_combined),
        save_dir      = 'results',
        plots_per_row = 5,
    )
    print("\nTop 5 worst features (highest KS statistic):")
    print(ks_results[['class', 'feature', 'ks_stat', 'p_value',
                       'wasserstein', 'similar']].head(5).to_string(index=False))

    # ── Correlation structure ─────────────────────────────────
    print("\n" + "=" * 60)
    print("CORRELATION STRUCTURE ANALYSIS")
    print("=" * 60)
    pipeline.plot_correlation(
        df_combined,
        n_synthetic      = len(df_combined),
        per_class        = True,
        save_path_prefix = "results/gmm_correlation"
    )

    # ── Save pipeline ─────────────────────────────────────────
    pipeline.save("gmm_pipeline_pytorch")
    print("\nDone!")
