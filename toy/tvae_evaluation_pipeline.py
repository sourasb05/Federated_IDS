"""
TVAE Synthetic Data Quality Evaluation Pipeline
================================================
1. Train a classifier on real data
2. Train a TVAE generator on real data (per-class)
3. Generate synthetic (fake) data from the generator
4. Evaluate the classifier using generated data (TRTS)
5. Correlation structure comparison (Real vs Synthetic)
6. Statistical significance tests

Based on: Xu et al. 2019 — "Modeling Tabular data using Conditional GAN"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import warnings
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# 1. MIN-MAX NORMALIZATION
# ============================================================

class ColumnMinMaxScaler:
    """
    Per-column min-max normalization to [0, 1].

    Why this matters:
      - `rank` is ~450-475, `dior` is 0-2, `diar` is 0-0.5
      - Without normalization, MSE loss is dominated by high-magnitude columns
      - GMM fits poorly on features with large absolute values
      - After normalization, all columns compete equally in the loss

    Stores min/max per column for inverse transform after generation.
    """

    def __init__(self):
        self.mins = {}
        self.maxs = {}
        self.columns = []

    def fit(self, df):
        self.columns = list(df.columns)
        for col in self.columns:
            vals = df[col].values.astype(np.float64)
            vals = vals[np.isfinite(vals)]
            self.mins[col] = float(vals.min()) if len(vals) > 0 else 0.0
            self.maxs[col] = float(vals.max()) if len(vals) > 0 else 1.0
            # Avoid division by zero for constant columns
            if self.maxs[col] == self.mins[col]:
                self.maxs[col] = self.mins[col] + 1.0
        return self

    def transform(self, df):
        out = df.copy()
        for col in self.columns:
            if col in out.columns:
                out[col] = (out[col] - self.mins[col]) / (self.maxs[col] - self.mins[col])
                out[col] = out[col].clip(0, 1)
        return out

    def inverse_transform(self, df):
        out = df.copy()
        for col in self.columns:
            if col in out.columns:
                out[col] = out[col] * (self.maxs[col] - self.mins[col]) + self.mins[col]
        return out


# ============================================================
# 2. DATA TRANSFORMER (Hybrid: Gaussian + GMM + Zero-inflated)
# ============================================================

class DataTransformer:
    """
    Hybrid per-column preprocessing with AUTO-DETECTION:

      - Dead columns (>99% zero):     dropped
      - Sparse columns (>50% zero):   zero-inflated [is_nonzero, normalized_value] (2 dims)
      - Gaussian columns (dense,      z-score normalized, single dim, NO tanh (1 dim)
        narrow-range, unimodal):       → decoder outputs raw value, loss is plain MSE
      - Multimodal dense columns:      Bayesian GMM mode-specific normalization (1+k dims)
      - Discrete columns:              one-hot encoding

    Gaussian detection:
      A column is 'gaussian' if it is dense (low zero_frac) AND
      the Bayesian GMM converges to exactly 1 active component.
      This means the data is unimodal — GMM encoding wastes parameters,
      and tanh activation causes variance collapse. Z-score + raw MSE is better.
    """

    def __init__(self, n_gmm_components=5, discrete_columns=None,
                 sparse_threshold=0.5, dead_threshold=0.99,
                 gaussian_columns=None):
        self.n_gmm_components = n_gmm_components
        self.discrete_columns = discrete_columns or []
        self.sparse_threshold = sparse_threshold
        self.dead_threshold = dead_threshold
        self.gaussian_columns = gaussian_columns  # None = auto-detect
        self.column_meta = []
        self._output_dim = 0
        self.dropped_columns = []

    def fit(self, df):
        self.column_meta = []
        self._output_dim = 0
        self.dropped_columns = []

        for col in df.columns:
            if col in self.discrete_columns:
                categories = sorted(df[col].unique())
                meta = {
                    'name': col, 'type': 'discrete',
                    'categories': categories,
                    'cat2idx': {c: i for i, c in enumerate(categories)},
                    'start': self._output_dim,
                    'dim': len(categories),
                }
                self._output_dim += len(categories)
                self.column_meta.append(meta)
                continue

            values = df[col].values.astype(np.float64)
            values = np.where(np.isfinite(values), values, 0.0)
            zero_frac = (values == 0).mean()

            # Dead column
            if zero_frac >= self.dead_threshold:
                self.dropped_columns.append(col)
                continue

            # Sparse column -> zero-inflated
            if zero_frac >= self.sparse_threshold:
                nz = values[values != 0]
                nz_mean = nz.mean() if len(nz) > 0 else 0.0
                nz_std = max(nz.std() if len(nz) > 1 else 1.0, 1e-8)
                meta = {
                    'name': col, 'type': 'zero_inflated',
                    'zero_frac': zero_frac, 'nz_mean': nz_mean, 'nz_std': nz_std,
                    'start': self._output_dim, 'dim': 2,
                }
                self._output_dim += 2
                self.column_meta.append(meta)
                continue

            # Dense column: check if user forced gaussian, or auto-detect
            force_gaussian = (self.gaussian_columns is not None and col in self.gaussian_columns)

            if not force_gaussian:
                # Auto-detect: fit GMM and check how many modes survive
                vals_2d = values.reshape(-1, 1)
                gmm = BayesianGaussianMixture(
                    n_components=self.n_gmm_components,
                    weight_concentration_prior=0.001,
                    max_iter=200, random_state=42, n_init=1
                )
                gmm.fit(vals_2d)
                active = gmm.weights_ > 0.01
                n_modes = max(int(active.sum()), 1)

                # If GMM finds only 1 mode -> column is unimodal -> use gaussian
                if n_modes == 1:
                    force_gaussian = True
                else:
                    # Multimodal -> use GMM encoding
                    meta = {
                        'name': col, 'type': 'continuous',
                        'gmm': gmm, 'active': active, 'n_modes': n_modes,
                        'means': gmm.means_.flatten()[active],
                        'stds': np.sqrt(gmm.covariances_.flatten()[active]),
                        'start': self._output_dim, 'dim': 1 + n_modes,
                    }
                    self._output_dim += 1 + n_modes
                    self.column_meta.append(meta)

            if force_gaussian:
                # Gaussian: z-score normalize, 1 output dim, no tanh
                col_mean = float(values.mean())
                col_std = max(float(values.std()), 1e-8)
                meta = {
                    'name': col, 'type': 'gaussian',
                    'mean': col_mean, 'std': col_std,
                    'start': self._output_dim, 'dim': 1,
                }
                self._output_dim += 1
                self.column_meta.append(meta)

        # Summary
        n_g = sum(1 for m in self.column_meta if m['type'] == 'gaussian')
        n_s = sum(1 for m in self.column_meta if m['type'] == 'zero_inflated')
        n_d = sum(1 for m in self.column_meta if m['type'] == 'continuous')
        n_disc = sum(1 for m in self.column_meta if m['type'] == 'discrete')
        if self.dropped_columns:
            print(f"    Dropped {len(self.dropped_columns)} dead columns")
        gauss_names = [m['name'] for m in self.column_meta if m['type'] == 'gaussian']
        print(f"    Encoding: {n_g} gaussian + {n_d} GMM + {n_s} sparse(ZI) + {n_disc} discrete "
              f"-> {self._output_dim} dims")
        if gauss_names:
            print(f"    Gaussian cols (z-score, no tanh): {gauss_names}")
        return self

    def transform(self, df):
        n = len(df)
        out = np.zeros((n, self._output_dim), dtype=np.float32)
        for meta in self.column_meta:
            s = meta['start']
            if meta['type'] == 'discrete':
                for i, val in enumerate(df[meta['name']].values):
                    out[i, s + meta['cat2idx'].get(val, 0)] = 1.0
            elif meta['type'] == 'zero_inflated':
                vals = np.where(np.isfinite(df[meta['name']].values), df[meta['name']].values, 0.0)
                out[:, s] = (vals != 0).astype(np.float32)
                out[:, s + 1] = np.where(vals != 0,
                    np.clip((vals - meta['nz_mean']) / (4 * meta['nz_std']), -1, 1), 0.0)
            elif meta['type'] == 'gaussian':
                vals = np.where(np.isfinite(df[meta['name']].values), df[meta['name']].values, 0.0)
                out[:, s] = (vals - meta['mean']) / meta['std']
            else:  # continuous (GMM)
                vals = np.where(np.isfinite(df[meta['name']].values), df[meta['name']].values, 0.0)
                vals_2d = vals.reshape(-1, 1)
                probs = meta['gmm'].predict_proba(vals_2d)[:, meta['active']]
                modes = probs.argmax(axis=1)
                for i in range(n):
                    k = modes[i]
                    out[i, s] = np.clip((vals[i] - meta['means'][k]) / (4 * meta['stds'][k] + 1e-8), -1, 1)
                    out[i, s + 1 + k] = 1.0
        return out

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        n = data.shape[0]
        result = {}
        for meta in self.column_meta:
            s = meta['start']
            if meta['type'] == 'discrete':
                idx = data[:, s:s + meta['dim']].argmax(axis=1)
                result[meta['name']] = [meta['categories'][min(i, len(meta['categories'])-1)] for i in idx]
            elif meta['type'] == 'zero_inflated':
                is_nz = data[:, s]
                normed = data[:, s + 1]
                result[meta['name']] = [
                    float(normed[i]) * 4 * meta['nz_std'] + meta['nz_mean'] if is_nz[i] > 0.5 else 0.0
                    for i in range(n)
                ]
            elif meta['type'] == 'gaussian':
                # Direct inverse z-score: value * std + mean
                z_vals = data[:, s]
                result[meta['name']] = [float(z_vals[i]) * meta['std'] + meta['mean'] for i in range(n)]
            else:  # continuous (GMM)
                alphas, betas = data[:, s], data[:, s+1:s+1+meta['n_modes']]
                modes = betas.argmax(axis=1)
                result[meta['name']] = [
                    float(alphas[i]) * 4 * meta['stds'][modes[i]] + meta['means'][modes[i]]
                    for i in range(n)
                ]
        for col in self.dropped_columns:
            result[col] = [0.0] * n
        df_out = pd.DataFrame(result)
        df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df_out.ffill().bfill().fillna(0.0)

    @property
    def output_dim(self):
        return self._output_dim


# ============================================================
# 2. TVAE MODEL
# ============================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers=2):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            prev = hidden_dim
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        layers = []
        prev = latent_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            prev = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class TVAE(nn.Module):
    """
    Tabular VAE with hybrid loss:
    
      - Gaussian columns:    direct MSE (NO tanh) — preserves variance
      - Dense continuous:    MSE(tanh(out), alpha) + CE(logits, mode)
      - Zero-inflated:       BCE(sigmoid(out), is_nonzero) + masked MSE(tanh(out), value)
      - Discrete:            CE(logits, category)
    """

    def __init__(self, input_dim, latent_dim=32, hidden_dim=256,
                 n_layers=2, transformer=None, loss_factor=2.0):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_layers)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.transformer = transformer
        self.loss_factor = loss_factor

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar

    def compute_loss(self, x, x_recon, mu, logvar, kl_weight=1.0):
        recon = torch.tensor(0.0, device=x.device)
        n_terms = 0
        for meta in self.transformer.column_meta:
            s = meta['start']
            if meta['type'] == 'gaussian':
                # Direct MSE — no tanh, no clipping, no mode selection
                # This lets the decoder freely output any z-score value
                recon += nn.functional.mse_loss(x_recon[:, s], x[:, s])
                n_terms += 1
            elif meta['type'] == 'continuous':
                nm = meta['n_modes']
                recon += nn.functional.mse_loss(torch.tanh(x_recon[:, s]), x[:, s])
                recon += nn.functional.cross_entropy(x_recon[:, s+1:s+1+nm], x[:, s+1:s+1+nm].argmax(1))
                n_terms += 2
            elif meta['type'] == 'zero_inflated':
                recon += nn.functional.binary_cross_entropy_with_logits(x_recon[:, s], x[:, s])
                mask = x[:, s]
                recon += nn.functional.mse_loss(torch.tanh(x_recon[:, s+1]) * mask, x[:, s+1] * mask)
                n_terms += 2
            else:
                recon += nn.functional.cross_entropy(x_recon[:, s:s+meta['dim']], x[:, s:s+meta['dim']].argmax(1))
                n_terms += 1
        recon = recon / max(n_terms, 1)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return self.loss_factor * recon + kl_weight * kl, recon, kl

    @torch.no_grad()
    def generate(self, n_samples):
        self.eval()
        raw = self.decoder(torch.randn(n_samples, self.latent_dim, device=DEVICE))
        out = raw.clone()
        for meta in self.transformer.column_meta:
            s = meta['start']
            if meta['type'] == 'gaussian':
                # Raw output — no activation, decoder outputs z-score directly
                # Adding small noise to prevent all samples collapsing to mean
                pass  # out[:, s] = raw[:, s] already set by clone
            elif meta['type'] == 'continuous':
                nm = meta['n_modes']
                out[:, s] = torch.tanh(raw[:, s])
                logits = raw[:, s+1:s+1+nm]
                gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
                modes = (logits + gumbel).argmax(dim=1)
                out[:, s+1:s+1+nm] = 0.0
                for i in range(n_samples):
                    out[i, s+1+modes[i]] = 1.0
            elif meta['type'] == 'zero_inflated':
                is_nz = (torch.sigmoid(raw[:, s]) > 0.5).float()
                out[:, s] = is_nz
                out[:, s+1] = torch.tanh(raw[:, s+1]) * is_nz
            else:
                logits = raw[:, s:s+meta['dim']]
                gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
                cats = (logits + gumbel).argmax(dim=1)
                out[:, s:s+meta['dim']] = 0.0
                for i in range(n_samples):
                    out[i, s+cats[i]] = 1.0
        self.train()
        return out

    def fit(self, data_np, epochs=300, batch_size=64, lr=1e-3,
            kl_anneal_epochs=100, kl_weight_max=0.05, verbose=True):
        """
        Train TVAE with KL annealing and capped KL weight.

        kl_weight_max: cap the KL weight to prevent posterior collapse.
            With loss_factor=4.0 and kl_weight_max=0.05, the balance is:
            L = 4.0 * recon + 0.05 * KL
            This keeps reconstruction dominant so the latent space stays informative.
        """
        self.to(DEVICE).train()
        loader = DataLoader(TensorDataset(torch.tensor(data_np, dtype=torch.float32)),
                            batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
        for epoch in range(epochs):
            kl_w = min(kl_weight_max, kl_weight_max * epoch / max(kl_anneal_epochs, 1))
            total, nb = 0, 0
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                x_r, mu, lv = self(batch)
                loss, rec, kl = self.compute_loss(batch, x_r, mu, lv, kl_w)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total += loss.item(); nb += 1
            scheduler.step()
            if verbose and (epoch+1) % 50 == 0:
                print(f"    Epoch {epoch+1:4d}/{epochs} | Loss: {total/nb:.4f} | "
                      f"Recon: {rec.item():.4f} | KL: {kl.item():.4f} | kl_w: {kl_w:.2f}")
        return self


# ============================================================
# 3. CORRELATION COMPARISON
# ============================================================

def plot_correlation_comparison(real_df, synth_df, feature_cols,
                                title="Correlation Structure", save_path=None):
    # Filter out constant columns to avoid NaN in correlation
    valid_cols = [c for c in feature_cols
                  if c in real_df.columns and c in synth_df.columns
                  and real_df[c].std() > 1e-10 and synth_df[c].std() > 1e-10]
    if len(valid_cols) < 2:
        print("  Not enough non-constant columns for correlation plot.")
        return

    real_corr = real_df[valid_cols].corr()
    synth_corr = synth_df[valid_cols].corr()
    diff_corr = (real_corr - synth_corr).abs()
    labels = [c[:8] for c in valid_cols]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    for ax, mat, cmap, vmin, vmax, ttl in [
        (axes[0], real_corr,  'RdBu_r', -1, 1, 'Real'),
        (axes[1], synth_corr, 'RdBu_r', -1, 1, 'Synthetic'),
        (axes[2], diff_corr,  'Reds',    0, 1, 'Difference (lower = better)'),
    ]:
        im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(ttl, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show(); plt.close()

    upper = np.triu_indices_from(diff_corr.values, k=1)
    print(f"  Mean |corr diff|: {diff_corr.values[upper].mean():.4f}")
    print(f"  Max  |corr diff|: {diff_corr.values[upper].max():.4f}")


# ============================================================
# 3.5 PER-FEATURE DISTRIBUTION COMPARISON
# ============================================================

def plot_distribution_comparison(real_df, synth_df, feature_cols, col_tests=None,
                                  label_col=None, save_path=None, max_cols=20):
    """
    Side-by-side distribution plots (histogram + KDE) for each feature.

    For each feature, plots:
      - Overlaid histograms (real vs synthetic)
      - KDE curves
      - Annotated with KS stat, p-value, JSD, Wasserstein

    If label_col is provided, also shows per-class breakdowns as
    separate rows of subplots.

    Args:
        real_df:       DataFrame with real data
        synth_df:      DataFrame with synthetic data
        feature_cols:  list of feature columns to compare
        col_tests:     DataFrame from compute_column_tests (optional, for annotations)
        label_col:     if provided, also plot per-class distributions
        save_path:     path to save the figure
        max_cols:      max features to plot (to avoid huge figures)
    """
    cols_to_plot = [c for c in feature_cols if c in real_df.columns and c in synth_df.columns]
    cols_to_plot = cols_to_plot[:max_cols]
    n_features = len(cols_to_plot)

    if n_features == 0:
        print("  No features to plot.")
        return

    # Build test lookup if provided
    test_lookup = {}
    if col_tests is not None and len(col_tests) > 0:
        for _, row in col_tests.iterrows():
            test_lookup[row['column']] = row

    # --- Overall comparison ---
    n_cols_grid = 4
    n_rows_grid = int(np.ceil(n_features / n_cols_grid))
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                              figsize=(5 * n_cols_grid, 4 * n_rows_grid))
    if n_rows_grid == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Feature Distributions: Real vs Synthetic",
                  fontsize=16, fontweight='bold', y=1.01)

    for idx, col in enumerate(cols_to_plot):
        row_i, col_i = idx // n_cols_grid, idx % n_cols_grid
        ax = axes[row_i, col_i]

        rv = real_df[col].dropna().values.astype(np.float64)
        sv = synth_df[col].dropna().values.astype(np.float64)

        # Determine if column is mostly zeros (sparse)
        zero_frac_real = (rv == 0).mean()
        zero_frac_synth = (sv == 0).mean()

        if zero_frac_real > 0.8:
            # Sparse column: show bar plot of zero vs nonzero fractions + nonzero histogram
            bar_x = [0, 1]
            bar_real = [zero_frac_real, 1 - zero_frac_real]
            bar_synth = [zero_frac_synth, 1 - zero_frac_synth]
            width = 0.35
            ax.bar([x - width/2 for x in bar_x], bar_real, width,
                   label='Real', color='steelblue', alpha=0.7)
            ax.bar([x + width/2 for x in bar_x], bar_synth, width,
                   label='Synthetic', color='coral', alpha=0.7)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Zero', 'Non-zero'])
            ax.set_ylabel('Fraction')
        else:
            # Dense column: overlaid histograms with KDE
            combined = np.concatenate([rv, sv])
            lo = np.percentile(combined, 1)
            hi = np.percentile(combined, 99)
            bins = np.linspace(lo, hi, 50)

            ax.hist(rv, bins=bins, density=True, alpha=0.5, color='steelblue', label='Real')
            ax.hist(sv, bins=bins, density=True, alpha=0.5, color='coral', label='Synthetic')

            # KDE overlay
            try:
                from scipy.stats import gaussian_kde
                if rv.std() > 0:
                    kde_r = gaussian_kde(rv)
                    x_kde = np.linspace(lo, hi, 200)
                    ax.plot(x_kde, kde_r(x_kde), color='steelblue', lw=2)
                if sv.std() > 0:
                    kde_s = gaussian_kde(sv)
                    x_kde = np.linspace(lo, hi, 200)
                    ax.plot(x_kde, kde_s(x_kde), color='coral', lw=2)
            except Exception:
                pass
            ax.set_ylabel('Density')

        # Title with test stats
        title = col
        if col in test_lookup:
            t = test_lookup[col]
            ks_tag = 'PASS' if t.get('ks_pass', False) else 'FAIL'
            title += f"\nKS={t['ks_stat']:.3f} (p={t['ks_pvalue']:.4f}) [{ks_tag}]"
            title += f"\nJSD={t['jsd']:.3f}  W={t['wasserstein']:.3f}"
        ax.set_title(title, fontsize=8, fontweight='bold',
                      color='green' if test_lookup.get(col, {}).get('ks_pass', False) else 'red')
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n_features, n_rows_grid * n_cols_grid):
        axes[idx // n_cols_grid, idx % n_cols_grid].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close()

    # --- Per-class comparison (if label provided) ---
    if label_col and label_col in real_df.columns and label_col in synth_df.columns:
        classes = sorted(real_df[label_col].unique())
        for cls in classes:
            rc = real_df[real_df[label_col] == cls]
            sc = synth_df[synth_df[label_col] == cls]
            if len(sc) < 5:
                continue

            fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                                      figsize=(5 * n_cols_grid, 4 * n_rows_grid))
            if n_rows_grid == 1:
                axes = axes.reshape(1, -1)
            fig.suptitle(f"Feature Distributions — Class {cls} "
                          f"(Real: {len(rc)}, Synthetic: {len(sc)})",
                          fontsize=14, fontweight='bold', y=1.01)

            for idx, col in enumerate(cols_to_plot):
                row_i, col_i = idx // n_cols_grid, idx % n_cols_grid
                ax = axes[row_i, col_i]

                rv = rc[col].dropna().values.astype(np.float64)
                sv = sc[col].dropna().values.astype(np.float64)

                zero_frac_real = (rv == 0).mean()

                if zero_frac_real > 0.8:
                    zfr, zfs = (rv == 0).mean(), (sv == 0).mean()
                    width = 0.35
                    ax.bar([-width/2, 1 - width/2], [zfr, 1 - zfr], width,
                           label='Real', color='steelblue', alpha=0.7)
                    ax.bar([width/2, 1 + width/2], [zfs, 1 - zfs], width,
                           label='Synthetic', color='coral', alpha=0.7)
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(['Zero', 'Non-zero'])
                    ax.set_ylabel('Fraction')
                else:
                    combined = np.concatenate([rv, sv])
                    lo, hi = np.percentile(combined, 1), np.percentile(combined, 99)
                    bins = np.linspace(lo, hi, 50)
                    ax.hist(rv, bins=bins, density=True, alpha=0.5,
                            color='steelblue', label='Real')
                    ax.hist(sv, bins=bins, density=True, alpha=0.5,
                            color='coral', label='Synthetic')
                    try:
                        from scipy.stats import gaussian_kde
                        if rv.std() > 0:
                            kde_r = gaussian_kde(rv)
                            x_k = np.linspace(lo, hi, 200)
                            ax.plot(x_k, kde_r(x_k), color='steelblue', lw=2)
                        if sv.std() > 0:
                            kde_s = gaussian_kde(sv)
                            x_k = np.linspace(lo, hi, 200)
                            ax.plot(x_k, kde_s(x_k), color='coral', lw=2)
                    except Exception:
                        pass
                    ax.set_ylabel('Density')

                ax.set_title(col, fontsize=9, fontweight='bold')
                ax.legend(fontsize=7, loc='upper right')
                ax.tick_params(labelsize=7)

            for idx in range(n_features, n_rows_grid * n_cols_grid):
                axes[idx // n_cols_grid, idx % n_cols_grid].set_visible(False)

            plt.tight_layout()
            cls_path = save_path.replace('.png', f'_class_{cls}.png') if save_path else None
            if cls_path:
                fig.savefig(cls_path, dpi=150, bbox_inches='tight')
                print(f"  Saved: {cls_path}")
            plt.show()
            plt.close()


# ============================================================
# 4. STATISTICAL SIGNIFICANCE TESTS
# ============================================================

def compute_column_tests(real_df, synth_df, feature_cols, alpha=0.05):
    """
    Per-column tests:
      - Kolmogorov-Smirnov:     H0 = same distribution
      - Mann-Whitney U:         H0 = same rank distribution
      - Wasserstein distance:   metric (lower = better)
      - Jensen-Shannon div:     metric [0,1] (0 = identical)
    """
    rows = []
    for col in feature_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        rv = real_df[col].dropna().values.astype(np.float64)
        sv = synth_df[col].dropna().values.astype(np.float64)
        if len(rv) < 2 or len(sv) < 2:
            continue

        ks_stat, ks_p = ks_2samp(rv, sv)
        try:
            mw_stat, mw_p = mannwhitneyu(rv, sv, alternative='two-sided')
        except ValueError:
            mw_stat, mw_p = np.nan, np.nan
        wd = wasserstein_distance(rv, sv)

        combined = np.concatenate([rv, sv])
        if combined.std() > 0:
            bins = np.histogram_bin_edges(combined, bins='auto')
            rh, _ = np.histogram(rv, bins=bins, density=True)
            sh, _ = np.histogram(sv, bins=bins, density=True)
            rh, sh = rh + 1e-10, sh + 1e-10
            rh, sh = rh / rh.sum(), sh / sh.sum()
            jsd = float(jensenshannon(rh, sh, base=2))
        else:
            jsd = 0.0

        rows.append({
            'column': col, 'ks_stat': ks_stat, 'ks_pvalue': ks_p,
            'mw_stat': mw_stat, 'mw_pvalue': mw_p,
            'wasserstein': wd, 'jsd': jsd,
        })
    return pd.DataFrame(rows)


def compute_mmd(X_real, X_synth, subsample=500, n_perm=500):
    """
    Maximum Mean Discrepancy — multivariate two-sample test.
    Uses RBF kernel with median heuristic + permutation test for p-value.
    """
    if len(X_real) > subsample:
        X_real = X_real[np.random.choice(len(X_real), subsample, replace=False)]
    if len(X_synth) > subsample:
        X_synth = X_synth[np.random.choice(len(X_synth), subsample, replace=False)]

    X_real, X_synth = np.float64(X_real), np.float64(X_synth)
    combined_sample = np.vstack([X_real[:100], X_synth[:100]])
    dists = np.sum((combined_sample[:, None] - combined_sample[None]) ** 2, axis=2)
    gamma = 1.0 / max(np.median(dists[dists > 0]), 1e-8)

    def rbf(A, B):
        sq = np.sum(A**2, 1, keepdims=True) + np.sum(B**2, 1, keepdims=True).T - 2 * A @ B.T
        return np.exp(-gamma * sq)

    def mmd2(X, Y):
        Kxx, Kyy, Kxy = rbf(X, X), rbf(Y, Y), rbf(X, Y)
        np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)
        n, m = len(X), len(Y)
        return Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.mean()

    observed = mmd2(X_real, X_synth)
    combined = np.vstack([X_real, X_synth])
    total = len(combined)
    n = len(X_real)
    perm_vals = []
    for _ in range(n_perm):
        p = np.random.permutation(total)
        perm_vals.append(mmd2(combined[p[:n]], combined[p[n:]]))
    p_value = (np.sum(np.array(perm_vals) >= observed) + 1) / (n_perm + 1)
    return float(observed), float(p_value)


def statistical_report(real_df, synth_df, feature_cols, label_col='label',
                        alpha=0.05, save_path=None):
    """
    Full statistical fidelity report with per-column tests, MMD, per-class breakdown,
    Bonferroni correction, and 4-panel visualization.
    """
    common = [c for c in feature_cols if c in real_df.columns and c in synth_df.columns]

    print(f"\n{'='*70}")
    print(f"STATISTICAL FIDELITY REPORT")
    print(f"{'='*70}")
    print(f"  Real: {len(real_df)}  |  Synthetic: {len(synth_df)}  |  "
          f"Features: {len(common)}  |  alpha = {alpha}")

    # --- Per-column tests ---
    ct = compute_column_tests(real_df, synth_df, common, alpha)
    n_cols = len(ct)
    bonf = alpha / max(n_cols, 1)
    ct['ks_pass'] = ct['ks_pvalue'] > bonf
    ct['mw_pass'] = ct['mw_pvalue'] > bonf

    ks_rate = ct['ks_pass'].mean() * 100
    mw_rate = ct['mw_pass'].mean() * 100
    m_jsd = ct['jsd'].mean()
    m_wd = ct['wasserstein'].mean()

    print(f"\n  Per-Column Tests (Bonferroni: alpha/{n_cols} = {bonf:.6f})")
    print(f"  {'─'*55}")
    print(f"  KS pass rate:          {ks_rate:5.1f}% ({int(ct['ks_pass'].sum())}/{n_cols})")
    print(f"  Mann-Whitney pass:     {mw_rate:5.1f}% ({int(ct['mw_pass'].sum())}/{n_cols})")
    print(f"  Mean JSD:              {m_jsd:.4f}   (0=identical, 1=max)")
    print(f"  Mean Wasserstein:      {m_wd:.4f}")

    # Worst 5 columns
    print(f"\n  Worst 5 by KS p-value:")
    for _, r in ct.nsmallest(5, 'ks_pvalue').iterrows():
        tag = "PASS" if r['ks_pass'] else "FAIL"
        print(f"    {r['column']:<15} D={r['ks_stat']:.4f}  p={r['ks_pvalue']:.6f}  [{tag}]")

    # --- Multivariate MMD ---
    Xr = real_df[common].values
    Xs = synth_df[common].values
    Xr = Xr[np.isfinite(Xr).all(axis=1)]
    Xs = Xs[np.isfinite(Xs).all(axis=1)]
    mmd_val, mmd_p = compute_mmd(Xr, Xs)
    mmd_pass = mmd_p > alpha

    print(f"\n  Multivariate MMD Test (joint distribution)")
    print(f"  {'─'*55}")
    print(f"  MMD^2 = {mmd_val:.6f}  |  p = {mmd_p:.4f}  |  {'PASS' if mmd_pass else 'FAIL'}")
    if mmd_pass:
        print(f"  -> Cannot reject H0: joint distributions are similar")
    else:
        print(f"  -> Reject H0: significant joint distribution difference")

    # --- Per-class ---
    class_results = {}
    if label_col in real_df.columns and label_col in synth_df.columns:
        print(f"\n  Per-Class Breakdown")
        print(f"  {'─'*55}")
        for cls in sorted(real_df[label_col].unique()):
            rc = real_df[real_df[label_col] == cls]
            sc = synth_df[synth_df[label_col] == cls]
            if len(sc) < 5:
                continue
            cct = compute_column_tests(rc, sc, common, alpha)
            n_c = len(cct)
            bonf_c = alpha / max(n_c, 1)
            cct['ks_pass'] = cct['ks_pvalue'] > bonf_c
            cct['mw_pass'] = cct['mw_pvalue'] > bonf_c
            Xrc = rc[common].values; Xsc = sc[common].values
            Xrc = Xrc[np.isfinite(Xrc).all(1)]; Xsc = Xsc[np.isfinite(Xsc).all(1)]
            c_mmd, c_mmd_p = compute_mmd(Xrc, Xsc)
            class_results[cls] = {
                'ks_rate': cct['ks_pass'].mean()*100, 'mw_rate': cct['mw_pass'].mean()*100,
                'jsd': cct['jsd'].mean(), 'mmd2': c_mmd, 'mmd_p': c_mmd_p,
            }
            print(f"  Class {cls} (real={len(rc)}, synth={len(sc)}): "
                  f"KS={cct['ks_pass'].mean()*100:.0f}%  MW={cct['mw_pass'].mean()*100:.0f}%  "
                  f"JSD={cct['jsd'].mean():.4f}  MMD p={c_mmd_p:.4f} [{'PASS' if c_mmd_p>alpha else 'FAIL'}]")

    # --- Full column table ---
    print(f"\n  {'Column':<15} {'KS-D':>7} {'KS-p':>10} {'KS':>5} "
          f"{'MW-p':>10} {'MW':>5} {'Wasser':>8} {'JSD':>7}")
    print(f"  {'-'*75}")
    for _, r in ct.iterrows():
        print(f"  {r['column']:<15} {r['ks_stat']:>7.4f} {r['ks_pvalue']:>10.6f} "
              f"{'PASS' if r['ks_pass'] else 'FAIL':>5} "
              f"{r['mw_pvalue']:>10.6f} {'PASS' if r['mw_pass'] else 'FAIL':>5} "
              f"{r['wasserstein']:>8.4f} {r['jsd']:>7.4f}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Statistical Fidelity: Real vs Synthetic", fontsize=14, fontweight='bold')

    # (a) KS p-values
    ax = axes[0, 0]
    pv = ct.sort_values('ks_pvalue')
    colors = ['green' if p else 'red' for p in pv['ks_pass']]
    ax.barh(range(len(pv)), -np.log10(pv['ks_pvalue'].clip(1e-300)), color=colors)
    ax.axvline(-np.log10(bonf), color='black', ls='--', label=f'Bonferroni ({bonf:.1e})')
    ax.set_yticks(range(len(pv)))
    ax.set_yticklabels([c[:10] for c in pv['column']], fontsize=6)
    ax.set_xlabel('-log10(p-value)')
    ax.set_title('KS Test (green=pass, red=fail)')
    ax.legend(fontsize=7)

    # (b) JSD
    ax = axes[0, 1]
    js = ct.sort_values('jsd', ascending=False)
    ax.barh(range(len(js)), js['jsd'],
            color=plt.cm.RdYlGn_r(js['jsd'].values / max(js['jsd'].max(), 0.01)))
    ax.set_yticks(range(len(js)))
    ax.set_yticklabels([c[:10] for c in js['column']], fontsize=6)
    ax.set_xlabel('Jensen-Shannon Divergence')
    ax.set_title('JSD per Column (lower = better)')

    # (c) Wasserstein
    ax = axes[1, 0]
    ws = ct.sort_values('wasserstein', ascending=False)
    ax.barh(range(len(ws)), ws['wasserstein'],
            color=plt.cm.RdYlGn_r(ws['wasserstein'].values / max(ws['wasserstein'].max(), 0.01)))
    ax.set_yticks(range(len(ws)))
    ax.set_yticklabels([c[:10] for c in ws['column']], fontsize=6)
    ax.set_xlabel('Wasserstein Distance')
    ax.set_title('Wasserstein per Column (lower = better)')

    # (d) Summary
    ax = axes[1, 1]
    ax.axis('off')
    txt = (f"SUMMARY\n{'─'*30}\n"
           f"alpha = {alpha}\n"
           f"Bonferroni = {bonf:.6f}\n\n"
           f"KS pass:       {ks_rate:.1f}%\n"
           f"Mann-Whitney:  {mw_rate:.1f}%\n"
           f"Mean JSD:      {m_jsd:.4f}\n"
           f"Mean Wasser:   {m_wd:.4f}\n"
           f"MMD^2:         {mmd_val:.6f}\n"
           f"MMD p-value:   {mmd_p:.4f}\n\n"
           f"{'─'*30}\n")
    if ks_rate >= 80 and mmd_pass:
        txt += "VERDICT: Strong fidelity\nSynthetic ~ Real"
        bc = '#d4edda'
    elif ks_rate >= 50:
        txt += "VERDICT: Moderate fidelity\nSome columns diverge"
        bc = '#fff3cd'
    else:
        txt += "VERDICT: Low fidelity\nSignificant gap"
        bc = '#f8d7da'
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=bc, alpha=0.8))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {save_path}")
    plt.show(); plt.close()

    return {
        'column_tests': ct, 'ks_pass_rate': ks_rate, 'mw_pass_rate': mw_rate,
        'mean_jsd': m_jsd, 'mean_wasserstein': m_wd,
        'mmd2': mmd_val, 'mmd_pvalue': mmd_p, 'mmd_pass': mmd_pass,
        'per_class': class_results,
    }


# ============================================================
# 4b. LIKELIHOOD FITNESS
# ============================================================

def likelihood_fitness(real_df, synth_df, feature_cols, label_col='label',
                       save_path=None):
    """
    Q1: Do columns in T_syn follow the same joint distribution as T_train?

    Approach: Gaussian Mixture Log-Likelihood Cross-Evaluation
      1. Fit GMM on REAL data → score REAL (LL_real_real) and SYNTHETIC (LL_real_synth)
      2. Fit GMM on SYNTHETIC data → score SYNTHETIC (LL_synth_synth) and REAL (LL_synth_real)
      3. Compare: if synthetic ≈ real distribution, then
           LL_real_synth ≈ LL_real_real   (synthetic looks likely under real model)
           LL_synth_real ≈ LL_synth_synth (real looks likely under synthetic model)

    Also computes a Likelihood Fitness Score (LFS):
      LFS = 1 - |LL_real_real - LL_real_synth| / |LL_real_real|
      LFS ∈ [0, 1], where 1 = perfect likelihood match
    """
    from sklearn.mixture import GaussianMixture

    common = [c for c in feature_cols if c in real_df.columns and c in synth_df.columns]
    X_real = real_df[common].values.astype(np.float64)
    X_synth = synth_df[common].values.astype(np.float64)

    # Clean NaN/inf
    X_real = X_real[np.isfinite(X_real).all(axis=1)]
    X_synth = X_synth[np.isfinite(X_synth).all(axis=1)]

    print(f"\n{'='*70}")
    print("LIKELIHOOD FITNESS: Do real and synthetic share the same distribution?")
    print(f"{'='*70}")

    # Determine optimal n_components via BIC on real data
    best_k, best_bic = 1, np.inf
    for k in range(1, min(10, len(X_real) // 10)):
        gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200)
        gmm.fit(X_real)
        bic = gmm.bic(X_real)
        if bic < best_bic:
            best_bic, best_k = bic, k
    print(f"  Optimal GMM components (BIC): {best_k}")

    # Fit GMM on real, score both
    gmm_real = GaussianMixture(n_components=best_k, random_state=42, max_iter=200)
    gmm_real.fit(X_real)
    ll_real_real = gmm_real.score(X_real)       # avg log-likelihood
    ll_real_synth = gmm_real.score(X_synth)

    # Fit GMM on synthetic, score both
    gmm_synth = GaussianMixture(n_components=best_k, random_state=42, max_iter=200)
    gmm_synth.fit(X_synth)
    ll_synth_synth = gmm_synth.score(X_synth)
    ll_synth_real = gmm_synth.score(X_real)

    # Likelihood Fitness Score
    lfs_real = 1 - abs(ll_real_real - ll_real_synth) / max(abs(ll_real_real), 1e-10)
    lfs_synth = 1 - abs(ll_synth_synth - ll_synth_real) / max(abs(ll_synth_synth), 1e-10)
    lfs = (lfs_real + lfs_synth) / 2
    lfs = max(0, min(1, lfs))  # clip to [0, 1]

    print(f"\n  {'Metric':<45} {'Value':>10}")
    print(f"  {'-'*57}")
    print(f"  {'GMM(real).score(real) = LL_rr':<45} {ll_real_real:>10.4f}")
    print(f"  {'GMM(real).score(synth) = LL_rs':<45} {ll_real_synth:>10.4f}")
    print(f"  {'GMM(synth).score(synth) = LL_ss':<45} {ll_synth_synth:>10.4f}")
    print(f"  {'GMM(synth).score(real) = LL_sr':<45} {ll_synth_real:>10.4f}")
    print(f"  {'-'*57}")
    print(f"  {'LL gap (real model): |LL_rr - LL_rs|':<45} {abs(ll_real_real - ll_real_synth):>10.4f}")
    print(f"  {'LL gap (synth model): |LL_ss - LL_sr|':<45} {abs(ll_synth_synth - ll_synth_real):>10.4f}")
    print(f"  {'Likelihood Fitness Score (LFS)':<45} {lfs:>10.4f}")

    # Interpretation
    print(f"\n  Interpretation:")
    if lfs > 0.9:
        print(f"  → Excellent: synthetic data has very similar joint density to real (LFS={lfs:.3f})")
    elif lfs > 0.7:
        print(f"  → Good: synthetic data captures main density structure (LFS={lfs:.3f})")
    elif lfs > 0.5:
        print(f"  → Moderate: notable density differences exist (LFS={lfs:.3f})")
    else:
        print(f"  → Poor: synthetic density significantly differs from real (LFS={lfs:.3f})")

    print(f"\n  What do the scores mean?")
    print(f"    LL_rr vs LL_rs: How likely is SYNTHETIC data under the REAL density model?")
    print(f"      If LL_rs ≈ LL_rr → synthetic samples land where real data is dense")
    print(f"      If LL_rs << LL_rr → synthetic samples land in low-density regions")
    print(f"    LL_ss vs LL_sr: How likely is REAL data under the SYNTHETIC density model?")
    print(f"      If LL_sr ≈ LL_ss → real samples are covered by synthetic density")
    print(f"      If LL_sr << LL_ss → real data has regions the generator misses")

    # Per-class analysis
    class_results = {}
    if label_col in real_df.columns and label_col in synth_df.columns:
        print(f"\n  Per-Class Likelihood Fitness:")
        print(f"  {'Class':<10} {'LL_rr':>10} {'LL_rs':>10} {'LL_ss':>10} {'LL_sr':>10} {'LFS':>8}")
        print(f"  {'-'*56}")
        for cls in sorted(real_df[label_col].unique()):
            rc = real_df[real_df[label_col] == cls][common].values.astype(np.float64)
            sc = synth_df[synth_df[label_col] == cls][common].values.astype(np.float64)
            rc = rc[np.isfinite(rc).all(1)]
            sc = sc[np.isfinite(sc).all(1)]
            if len(rc) < 10 or len(sc) < 10:
                continue
            k_cls = max(1, min(best_k, len(rc) // 10))
            g_r = GaussianMixture(n_components=k_cls, random_state=42).fit(rc)
            g_s = GaussianMixture(n_components=k_cls, random_state=42).fit(sc)
            c_ll_rr, c_ll_rs = g_r.score(rc), g_r.score(sc)
            c_ll_ss, c_ll_sr = g_s.score(sc), g_s.score(rc)
            c_lfs_r = 1 - abs(c_ll_rr - c_ll_rs) / max(abs(c_ll_rr), 1e-10)
            c_lfs_s = 1 - abs(c_ll_ss - c_ll_sr) / max(abs(c_ll_ss), 1e-10)
            c_lfs = max(0, min(1, (c_lfs_r + c_lfs_s) / 2))
            class_results[cls] = {'ll_rr': c_ll_rr, 'll_rs': c_ll_rs,
                                   'll_ss': c_ll_ss, 'll_sr': c_ll_sr, 'lfs': c_lfs}
            print(f"  {cls:<10} {c_ll_rr:>10.4f} {c_ll_rs:>10.4f} "
                  f"{c_ll_ss:>10.4f} {c_ll_sr:>10.4f} {c_lfs:>8.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Likelihood Fitness: Real vs Synthetic", fontsize=14, fontweight='bold')

    # (a) Cross log-likelihoods
    ax = axes[0]
    labels_bar = ['GMM(real)\non real', 'GMM(real)\non synth', 'GMM(synth)\non synth', 'GMM(synth)\non real']
    values = [ll_real_real, ll_real_synth, ll_synth_synth, ll_synth_real]
    colors_bar = ['steelblue', 'coral', 'coral', 'steelblue']
    bars = ax.bar(range(4), values, color=colors_bar, alpha=0.8, edgecolor='gray')
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_bar, fontsize=8)
    ax.set_ylabel('Avg Log-Likelihood')
    ax.set_title('Cross Log-Likelihood Scores')
    # Add value labels
    for bar_item, val in zip(bars, values):
        ax.text(bar_item.get_x() + bar_item.get_width()/2, bar_item.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # (b) LFS gauge
    ax = axes[1]
    ax.axis('off')
    color_lfs = '#d4edda' if lfs > 0.9 else '#fff3cd' if lfs > 0.7 else '#f8d7da'
    txt = (f"LIKELIHOOD FITNESS SCORE\n{'─'*30}\n\n"
           f"Overall LFS:  {lfs:.4f}\n\n"
           f"LL gap (real model):   {abs(ll_real_real - ll_real_synth):.4f}\n"
           f"LL gap (synth model):  {abs(ll_synth_synth - ll_synth_real):.4f}\n\n"
           f"{'─'*30}\n"
           f"1.0 = identical densities\n"
           f"0.0 = completely different\n")
    ax.text(0.1, 0.95, txt, transform=ax.transAxes, fontsize=11,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=color_lfs, alpha=0.8))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {save_path}")
    plt.show(); plt.close()

    return {
        'll_real_real': ll_real_real, 'll_real_synth': ll_real_synth,
        'll_synth_synth': ll_synth_synth, 'll_synth_real': ll_synth_real,
        'lfs': lfs, 'per_class': class_results,
    }


# ============================================================
# 4c. ML EFFICACY (TSTR + TRTS)
# ============================================================

def ml_efficacy(real_train, real_test, synthetic, feature_cols, label_col='label',
                save_path=None):
    """
    Q2: Can a model trained on T_syn achieve similar performance on T_test
        as a model trained on T_train?

    Evaluations:
      1. TRTR: Train on Real, Test on Real       → baseline accuracy
      2. TSTR: Train on Synthetic, Test on Real   → ML efficacy (the key metric)
      3. TRTS: Train on Real, Test on Synthetic   → synthetic recognizability
      4. TSTS: Train on Synthetic, Test on Synthetic → synthetic self-consistency

    ML Efficacy Score = TSTR / TRTR
      1.0 = synthetic is as good as real for training
      <1.0 = training on synthetic loses information

    Runs with multiple classifiers for robustness:
      - Random Forest
      - Logistic Regression
      - K-Nearest Neighbors
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    common = [c for c in feature_cols if c in real_train.columns
              and c in real_test.columns and c in synthetic.columns]

    X_real_train = real_train[common].values
    y_real_train = real_train[label_col].values
    X_real_test = real_test[common].values
    y_real_test = real_test[label_col].values
    X_synth = synthetic[common].values
    y_synth = synthetic[label_col].values

    print(f"\n{'='*70}")
    print("ML EFFICACY: Can a model trained on synthetic match real-trained?")
    print(f"{'='*70}")
    print(f"  Real train: {len(X_real_train)} | Real test: {len(X_real_test)} | "
          f"Synthetic: {len(X_synth)}")

    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'KNN (k=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ]),
    }

    results = {}

    print(f"\n  {'Classifier':<22} {'TRTR':>7} {'TSTR':>7} {'TRTS':>7} {'TSTS':>7} {'Efficacy':>9}")
    print(f"  {'-'*63}")

    for name, clf_template in classifiers.items():
        from sklearn.base import clone

        # TRTR: Train Real, Test Real (baseline)
        clf_rr = clone(clf_template).fit(X_real_train, y_real_train)
        trtr = accuracy_score(y_real_test, clf_rr.predict(X_real_test))

        # TSTR: Train Synthetic, Test Real (THE key metric)
        clf_sr = clone(clf_template).fit(X_synth, y_synth)
        tstr = accuracy_score(y_real_test, clf_sr.predict(X_real_test))

        # TRTS: Train Real, Test Synthetic
        trts = accuracy_score(y_synth, clf_rr.predict(X_synth))

        # TSTS: Train Synthetic, Test Synthetic
        tsts = accuracy_score(y_synth, clf_sr.predict(X_synth))

        # Efficacy score
        efficacy = tstr / trtr if trtr > 0 else 0

        results[name] = {
            'trtr': trtr, 'tstr': tstr, 'trts': trts, 'tsts': tsts,
            'efficacy': efficacy,
        }

        print(f"  {name:<22} {trtr:>7.4f} {tstr:>7.4f} {trts:>7.4f} {tsts:>7.4f} {efficacy:>9.4f}")

    # Averages
    avg_trtr = np.mean([r['trtr'] for r in results.values()])
    avg_tstr = np.mean([r['tstr'] for r in results.values()])
    avg_trts = np.mean([r['trts'] for r in results.values()])
    avg_tsts = np.mean([r['tsts'] for r in results.values()])
    avg_eff = np.mean([r['efficacy'] for r in results.values()])

    print(f"  {'-'*63}")
    print(f"  {'AVERAGE':<22} {avg_trtr:>7.4f} {avg_tstr:>7.4f} {avg_trts:>7.4f} {avg_tsts:>7.4f} {avg_eff:>9.4f}")

    # Interpretation
    print(f"\n  Interpretation:")
    print(f"    TRTR = {avg_trtr:.4f} (baseline: real→real)")
    print(f"    TSTR = {avg_tstr:.4f} (key metric: synthetic→real)")
    print(f"    Efficacy = TSTR/TRTR = {avg_eff:.4f}")
    if avg_eff > 0.95:
        print(f"    → Excellent: synthetic data nearly as useful as real for training")
    elif avg_eff > 0.85:
        print(f"    → Good: synthetic captures most discriminative information")
    elif avg_eff > 0.70:
        print(f"    → Moderate: some discriminative information lost in generation")
    else:
        print(f"    → Poor: synthetic data inadequate for training")

    print(f"\n  What each metric means:")
    print(f"    TRTR: Upper bound — best achievable with real data")
    print(f"    TSTR: Can synthetic REPLACE real for training? (main question)")
    print(f"    TRTS: Does the real model recognize synthetic as valid?")
    print(f"    TSTS: Is synthetic data internally consistent?")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ML Efficacy: Real vs Synthetic Training", fontsize=14, fontweight='bold')

    # (a) Grouped bar chart
    ax = axes[0]
    x = np.arange(len(classifiers))
    width = 0.2
    metrics = ['trtr', 'tstr', 'trts', 'tsts']
    labels_m = ['TRTR\n(real→real)', 'TSTR\n(synth→real)', 'TRTS\n(real→synth)', 'TSTS\n(synth→synth)']
    colors_m = ['steelblue', 'coral', '#66c2a5', '#fc8d62']
    clf_names = list(results.keys())

    for i, (metric, label, color) in enumerate(zip(metrics, labels_m, colors_m)):
        vals = [results[n][metric] for n in clf_names]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar_item, val in zip(bars, vals):
            ax.text(bar_item.get_x() + bar_item.get_width()/2, bar_item.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([n[:12] for n in clf_names], fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Accuracy by Classifier and Training Source')

    # (b) Efficacy summary
    ax = axes[1]
    ax.axis('off')
    color_eff = '#d4edda' if avg_eff > 0.95 else '#fff3cd' if avg_eff > 0.85 else '#f8d7da'
    txt = (f"ML EFFICACY SUMMARY\n{'─'*35}\n\n"
           f"{'Metric':<20} {'Score':>8}\n"
           f"{'─'*35}\n")
    for name in clf_names:
        r = results[name]
        txt += f"{name[:18]:<20} {r['efficacy']:>8.4f}\n"
    txt += (f"{'─'*35}\n"
            f"{'AVERAGE EFFICACY':<20} {avg_eff:>8.4f}\n\n"
            f"{'─'*35}\n"
            f"Efficacy = TSTR / TRTR\n"
            f"1.0 = synthetic = real for training\n"
            f"Goal: > 0.95\n")
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=color_eff, alpha=0.8))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {save_path}")
    plt.show(); plt.close()

    return {
        'per_classifier': results,
        'avg_trtr': avg_trtr, 'avg_tstr': avg_tstr,
        'avg_trts': avg_trts, 'avg_tsts': avg_tsts,
        'avg_efficacy': avg_eff,
    }


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def run_pipeline(csv_path,
                 label_column='label',
                 latent_dim=24,
                 hidden_dim=128,
                 n_layers=2,
                 epochs=300,
                 kl_anneal_epochs=100,
                 kl_weight_max=0.05,
                 loss_factor=4.0,
                 lr=1e-3,
                 batch_size=64,
                 n_gmm_components=3,
                 sparse_threshold=0.5,
                 dead_threshold=0.99,
                 gaussian_columns=None,
                 test_size=0.2,
                 alpha=0.05,
                 output_dir='results'):
    """
    Full pipeline:
      1. Load data, train/test split
      2. Train classifier on real training data
      3. Train per-class TVAE generators on real training data
      4. Generate synthetic data
      5. Evaluate classifier on synthetic data (TRTS)
      6. Correlation comparison plots
      7. Statistical significance tests
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- Load & prepare ----
    print("=" * 70)
    print("STEP 1: Load Data")
    print("=" * 70)
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    feature_cols = [c for c in df.columns if c != label_column]
    print(f"  Samples: {len(df)}  |  Features: {len(feature_cols)}  |  Label: {label_column}")
    print(f"  Label distribution: {dict(df[label_column].value_counts())}")

    # Sparsity
    zf = (df[feature_cols] == 0).mean()
    print(f"\n  Sparsity: {(zf > 0.99).sum()} dead, "
          f"{((zf > sparse_threshold) & (zf <= 0.99)).sum()} sparse, "
          f"{(zf <= sparse_threshold).sum()} dense")

    # Train/test split
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=test_size,
                                          stratify=df[label_column], random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f"  Train: {len(df_train)}  |  Test: {len(df_test)}")

    # ---- Train classifier on real data ----
    print(f"\n{'='*70}")
    print("STEP 2: Train Classifier on Real Data")
    print("=" * 70)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(df_train[feature_cols].values, df_train[label_column].values)

    train_acc = accuracy_score(df_train[label_column], clf.predict(df_train[feature_cols].values))
    test_acc = accuracy_score(df_test[label_column], clf.predict(df_test[feature_cols].values))
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"\n  Test Report:\n{classification_report(df_test[label_column], clf.predict(df_test[feature_cols].values))}")

    # ---- Train per-class TVAE generators on real data ----
    print(f"{'='*70}")
    print("STEP 3: Train Per-Class TVAE Generators (Hybrid Encoding)")
    print("=" * 70)
    print("  Encoding strategy:")
    print("    - Unimodal dense columns (e.g. rank): z-score, NO tanh -> preserves variance")
    print("    - Multimodal dense columns:           GMM mode-specific + tanh")
    print("    - Sparse columns (>50% zero):         zero-inflated [is_nz, value]")
    print("    - Dead columns (>99% zero):           dropped")
    print(f"  Settings: loss_factor={loss_factor}, kl_weight_max={kl_weight_max}, epochs={epochs}")
    if gaussian_columns:
        print(f"  Forced gaussian: {gaussian_columns}")
    classes = sorted(df_train[label_column].unique())
    generators = {}     # class -> (tvae, transformer)

    for cls_label in classes:
        cls_data = df_train[df_train[label_column] == cls_label][feature_cols]
        print(f"\n  Class {cls_label} ({len(cls_data)} samples)")

        # Hybrid transformer: auto-detects gaussian vs GMM vs zero-inflated
        transformer = DataTransformer(
            n_gmm_components=n_gmm_components,
            discrete_columns=[],
            sparse_threshold=sparse_threshold,
            dead_threshold=dead_threshold,
            gaussian_columns=gaussian_columns,
        )
        transformer.fit(cls_data)
        transformed = transformer.transform(cls_data)

        tvae = TVAE(
            input_dim=transformed.shape[1],
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            transformer=transformer,
            loss_factor=loss_factor,
        )
        tvae.fit(transformed, epochs=epochs, batch_size=batch_size,
                 lr=lr, kl_anneal_epochs=kl_anneal_epochs,
                 kl_weight_max=kl_weight_max)
        generators[cls_label] = (tvae, transformer)

    # ---- Generate synthetic data ----
    print(f"\n{'='*70}")
    print("STEP 4: Generate Synthetic Data")
    print("=" * 70)
    n_per_class = len(df_train) // len(classes)
    synth_parts = []
    for cls_label in classes:
        tvae, transformer = generators[cls_label]
        encoded = tvae.generate(n_per_class)
        synth_df = transformer.inverse_transform(encoded)
        synth_df[label_column] = cls_label
        synth_parts.append(synth_df)
        print(f"  Class {cls_label}: generated {n_per_class} samples")
    synthetic = pd.concat(synth_parts, ignore_index=True)
    print(f"  Total synthetic: {len(synthetic)}")
    print(f"  Synthetic label dist: {dict(synthetic[label_column].value_counts())}")

    # Save synthetic data
    synth_path = os.path.join(output_dir, 'synthetic_data.csv')
    synthetic.to_csv(synth_path, index=False)
    print(f"  Saved to: {synth_path}")

    # ---- Evaluate classifier on synthetic data (TRTS) ----
    print(f"\n{'='*70}")
    print("STEP 5: Evaluate Classifier on Synthetic Data (TRTS)")
    print("=" * 70)
    synth_feat_cols = [c for c in feature_cols if c in synthetic.columns]
    X_synth = synthetic[synth_feat_cols].values
    y_synth = synthetic[label_column].values
    trts_acc = accuracy_score(y_synth, clf.predict(X_synth))

    print(f"  Classifier trained on:  REAL data ({len(df_train)} samples)")
    print(f"  Classifier tested on:   SYNTHETIC data ({len(synthetic)} samples)")
    print(f"\n  TRTS accuracy: {trts_acc:.4f}")
    print(f"  (compare to test-on-real accuracy: {test_acc:.4f})")
    print(f"\n  TRTS Report:\n{classification_report(y_synth, clf.predict(X_synth))}")

    # Interpret
    gap = abs(trts_acc - test_acc)
    if gap < 0.05:
        print(f"  -> Excellent: TRTS within 5% of real test ({gap:.1%} gap)")
    elif gap < 0.15:
        print(f"  -> Moderate: TRTS within 15% of real test ({gap:.1%} gap)")
    else:
        print(f"  -> Poor: TRTS differs significantly from real test ({gap:.1%} gap)")

    # ---- Correlation comparison ----
    print(f"\n{'='*70}")
    print("STEP 6: Correlation Structure Comparison")
    print("=" * 70)

    common_cols = [c for c in feature_cols if c in synthetic.columns]

    # Overall
    print("\n  [Overall]")
    plot_correlation_comparison(df_train, synthetic, common_cols,
                                title="Correlation: Real vs Synthetic (Overall)",
                                save_path=os.path.join(output_dir, 'correlation_overall.png'))

    # Per class
    for cls_label in classes:
        real_cls = df_train[df_train[label_column] == cls_label]
        synth_cls = synthetic[synthetic[label_column] == cls_label]
        print(f"\n  [Class {cls_label}]")
        plot_correlation_comparison(
            real_cls, synth_cls, common_cols,
            title=f"Correlation: Class {cls_label}",
            save_path=os.path.join(output_dir, f'correlation_class_{cls_label}.png')
        )

    # ---- Statistical significance ----
    print(f"\n{'='*70}")
    print("STEP 7: Statistical Significance Tests")
    print("=" * 70)
    report = statistical_report(
        df_train, synthetic, common_cols,
        label_col=label_column, alpha=alpha,
        save_path=os.path.join(output_dir, 'statistical_report.png')
    )

    # ---- Per-feature distribution comparison ----
    print(f"\n{'='*70}")
    print("STEP 8: Per-Feature Distribution Comparison")
    print("=" * 70)
    plot_distribution_comparison(
        df_train, synthetic, common_cols,
        col_tests=report['column_tests'],
        label_col=label_column,
        save_path=os.path.join(output_dir, 'distribution_comparison.png'),
    )

    # ---- Likelihood fitness ----
    print(f"\n{'='*70}")
    print("STEP 9: Likelihood Fitness (Joint Distribution)")
    print("=" * 70)
    lf_report = likelihood_fitness(
        df_train, synthetic, common_cols,
        label_col=label_column,
        save_path=os.path.join(output_dir, 'likelihood_fitness.png'),
    )

    # ---- ML efficacy ----
    print(f"\n{'='*70}")
    print("STEP 10: ML Efficacy (TSTR vs TRTR)")
    print("=" * 70)
    ml_report = ml_efficacy(
        df_train, df_test, synthetic, common_cols,
        label_col=label_column,
        save_path=os.path.join(output_dir, 'ml_efficacy.png'),
    )

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Real train samples:       {len(df_train)}")
    print(f"  Real test samples:        {len(df_test)}")
    print(f"  Synthetic samples:        {len(synthetic)}")
    print(f"\n  --- Classification ---")
    print(f"  Classifier test acc:      {test_acc:.4f}")
    print(f"  TRTS acc (test on synth): {trts_acc:.4f}")
    print(f"\n  --- Statistical Fidelity ---")
    print(f"  KS pass rate:             {report['ks_pass_rate']:.1f}%")
    print(f"  Mann-Whitney pass:        {report['mw_pass_rate']:.1f}%")
    print(f"  Mean JSD:                 {report['mean_jsd']:.4f}")
    print(f"  Mean Wasserstein:         {report['mean_wasserstein']:.4f}")
    print(f"  MMD^2 (joint):            {report['mmd2']:.6f} (p={report['mmd_pvalue']:.4f})")
    print(f"\n  --- Likelihood Fitness (Q1) ---")
    print(f"  LFS (overall):            {lf_report['lfs']:.4f}")
    print(f"  LL gap (real model):      {abs(lf_report['ll_real_real'] - lf_report['ll_real_synth']):.4f}")
    print(f"  LL gap (synth model):     {abs(lf_report['ll_synth_synth'] - lf_report['ll_synth_real']):.4f}")
    print(f"\n  --- ML Efficacy (Q2) ---")
    print(f"  TRTR (real→real):         {ml_report['avg_trtr']:.4f}")
    print(f"  TSTR (synth→real):        {ml_report['avg_tstr']:.4f}")
    print(f"  Efficacy (TSTR/TRTR):     {ml_report['avg_efficacy']:.4f}")
    print(f"\n  All outputs in:           {output_dir}/")

    return {
        'classifier': clf,
        'generators': generators,
        'real_train': df_train, 'real_test': df_test,
        'synthetic': synthetic,
        'test_acc': test_acc, 'trts_acc': trts_acc,
        'statistical_report': report,
        'likelihood_fitness': lf_report,
        'ml_efficacy': ml_report,
    }


# ============================================================
# 6. RUN
# ============================================================

if __name__ == "__main__":
    results = run_pipeline(
        csv_path="/Users/souba636/Documents/vinnova_paper_2/Federated_IDS/attack_data/localrepair_var10_dec/1_features_timeseries_60_sec.csv",
        label_column='label',
        latent_dim=24,
        hidden_dim=128,
        n_layers=2,
        epochs=500,
        kl_anneal_epochs=100,
        kl_weight_max=0.05,        # cap KL weight to prevent posterior collapse
        loss_factor=4.0,           # stronger reconstruction pressure (was 2.0)
        lr=1e-3,
        batch_size=64,
        n_gmm_components=3,
        sparse_threshold=0.5,
        dead_threshold=0.99,
        gaussian_columns=['rank', 'rank.1'],  # force z-score + no tanh for these
        test_size=0.2,
        alpha=0.05,
        output_dir='results',
    )
    print("\nDone!")