"""
TVAE-based Generative Replay Pipeline for Continual Learning
=============================================================
PyTorch implementation

Key improvements over NumPy version:
  - Autograd handles all backpropagation
  - Per-class TVAE generators (fixes label collapse)
  - Proper DataLoader batching
  - GPU support if available

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
import warnings
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# 1. DATA TRANSFORMER (GMM-based mode-specific normalization)
# ============================================================

class TVAEDataTransformer:
    """
    Mode-specific normalization from the TVAE paper.
    
    For each continuous column:
      - Fit Bayesian GMM with k components
      - Encode each value as:
          alpha_i = (x - mu_k) / (4 * sigma_k)   [normalized scalar in ~[-1,1]]
          beta_i  = one-hot mode indicator          [which Gaussian mode]
    
    For discrete columns:
      - One-hot encode
    """

    def __init__(self, n_gmm_components=5, discrete_columns=None):
        self.n_gmm_components = n_gmm_components
        self.discrete_columns = discrete_columns or []
        self.column_meta = []   # ordered list of column metadata
        self._output_dim = 0

    def fit(self, df):
        self.column_meta = []
        self._output_dim = 0

        for col in df.columns:
            if col in self.discrete_columns:
                categories = sorted(df[col].unique())
                meta = {
                    'name': col,
                    'type': 'discrete',
                    'categories': categories,
                    'cat2idx': {c: i for i, c in enumerate(categories)},
                    'start': self._output_dim,
                    'dim': len(categories),
                }
                self._output_dim += len(categories)
            else:
                values = df[col].values.reshape(-1, 1).astype(np.float64)
                # Clean NaN/inf before fitting GMM
                mask = np.isfinite(values.flatten())
                if not mask.all():
                    col_median = np.nanmedian(values)
                    values[~mask.reshape(-1, 1)] = col_median if np.isfinite(col_median) else 0.0
                gmm = BayesianGaussianMixture(
                    n_components=self.n_gmm_components,
                    weight_concentration_prior=0.001,
                    max_iter=200, random_state=42, n_init=1
                )
                gmm.fit(values)

                active = gmm.weights_ > 0.01
                n_modes = max(int(active.sum()), 1)

                meta = {
                    'name': col,
                    'type': 'continuous',
                    'gmm': gmm,
                    'active': active,
                    'n_modes': n_modes,
                    'means': gmm.means_.flatten()[active],
                    'stds': np.sqrt(gmm.covariances_.flatten()[active]),
                    'start': self._output_dim,
                    'dim': 1 + n_modes,         # alpha + beta one-hot
                }
                self._output_dim += 1 + n_modes

            self.column_meta.append(meta)
        return self

    def transform(self, df):
        n = len(df)
        out = np.zeros((n, self._output_dim), dtype=np.float32)

        for meta in self.column_meta:
            s = meta['start']

            if meta['type'] == 'discrete':
                for i, val in enumerate(df[meta['name']].values):
                    out[i, s + meta['cat2idx'].get(val, 0)] = 1.0
            else:
                vals = df[meta['name']].values.reshape(-1, 1).astype(np.float64)
                # Clean NaN/inf before GMM predict
                mask = np.isfinite(vals.flatten())
                if not mask.all():
                    col_median = np.nanmedian(vals)
                    vals[~mask.reshape(-1, 1)] = col_median if np.isfinite(col_median) else 0.0
                probs = meta['gmm'].predict_proba(vals)[:, meta['active']]
                modes = probs.argmax(axis=1)

                for i in range(n):
                    k = modes[i]
                    alpha = (vals[i, 0] - meta['means'][k]) / (4 * meta['stds'][k] + 1e-8)
                    out[i, s] = np.clip(alpha, -1, 1)
                    out[i, s + 1 + k] = 1.0

        return out

    def inverse_transform(self, data):
        """Convert TVAE output back to original feature space."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        n = data.shape[0]
        result = {}

        for meta in self.column_meta:
            s = meta['start']

            if meta['type'] == 'discrete':
                indices = data[:, s:s + meta['dim']].argmax(axis=1)
                cats = meta['categories']
                result[meta['name']] = [cats[min(idx, len(cats) - 1)] for idx in indices]
            else:
                alphas = data[:, s]
                betas = data[:, s + 1: s + 1 + meta['n_modes']]
                modes = betas.argmax(axis=1)
                vals = []
                for i in range(n):
                    k = modes[i]
                    x = alphas[i] * 4 * meta['stds'][k] + meta['means'][k]
                    vals.append(float(x))
                result[meta['name']] = vals

        df_out = pd.DataFrame(result)
        # Clean any NaN/inf produced by decoder extremes
        df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_out = df_out.ffill().bfill().fillna(0.0)
        return df_out

    @property
    def output_dim(self):
        return self._output_dim


# ============================================================
# 2. TVAE MODEL (PyTorch)
# ============================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class TVAE(nn.Module):
    """
    Tabular VAE.

    Loss = loss_factor * Recon_loss + KL_loss

    Reconstruction is column-aware:
      - continuous alpha:  MSE on tanh(output)
      - continuous beta:   CrossEntropy on mode logits
      - discrete:          CrossEntropy on category logits
    """

    def __init__(self, input_dim, latent_dim=16, hidden_dim=128,
                 transformer=None, loss_factor=2.0):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.transformer = transformer
        self.loss_factor = loss_factor
        self.train_losses = []

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def compute_loss(self, x, x_recon, mu, logvar):
        """Column-aware reconstruction + KL divergence."""
        recon = torch.tensor(0.0, device=x.device)

        if self.transformer is not None:
            for meta in self.transformer.column_meta:
                s = meta['start']

                if meta['type'] == 'continuous':
                    nm = meta['n_modes']

                    # Alpha: MSE (through tanh activation)
                    alpha_true = x[:, s]
                    alpha_pred = torch.tanh(x_recon[:, s])
                    recon = recon + nn.functional.mse_loss(alpha_pred, alpha_true)

                    # Beta: cross-entropy on mode logits
                    beta_true = x[:, s + 1: s + 1 + nm].argmax(dim=1)
                    beta_logits = x_recon[:, s + 1: s + 1 + nm]
                    recon = recon + nn.functional.cross_entropy(beta_logits, beta_true)

                else:  # discrete
                    true_idx = x[:, s: s + meta['dim']].argmax(dim=1)
                    logits = x_recon[:, s: s + meta['dim']]
                    recon = recon + nn.functional.cross_entropy(logits, true_idx)
        else:
            recon = nn.functional.mse_loss(x_recon, x)

        # KL divergence
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return self.loss_factor * recon + kl, recon, kl

    @torch.no_grad()
    def generate(self, n_samples):
        """Sample z ~ N(0,I), decode, apply activations per column."""
        self.eval()
        z = torch.randn(n_samples, self.latent_dim, device=DEVICE)
        raw = self.decoder(z)
        out = raw.clone()

        if self.transformer is not None:
            for meta in self.transformer.column_meta:
                s = meta['start']

                if meta['type'] == 'continuous':
                    nm = meta['n_modes']
                    out[:, s] = torch.tanh(raw[:, s])

                    # Gumbel-argmax for crisp mode selection
                    logits = raw[:, s + 1: s + 1 + nm]
                    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
                    modes = (logits + gumbel).argmax(dim=1)
                    out[:, s + 1: s + 1 + nm] = 0.0
                    for i in range(n_samples):
                        out[i, s + 1 + modes[i]] = 1.0
                else:
                    logits = raw[:, s: s + meta['dim']]
                    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
                    cats = (logits + gumbel).argmax(dim=1)
                    out[:, s: s + meta['dim']] = 0.0
                    for i in range(n_samples):
                        out[i, s + cats[i]] = 1.0

        self.train()
        return out

    def fit(self, data_np, epochs=300, batch_size=64, lr=1e-3, verbose=True):
        """Train the TVAE end-to-end."""
        self.to(DEVICE)
        self.train()

        dataset = TensorDataset(torch.tensor(data_np, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        self.train_losses = []
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                x_recon, mu, logvar = self(batch)
                loss, recon, kl = self.compute_loss(batch, x_recon, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg = total_loss / max(n_batches, 1)
            self.train_losses.append(avg)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1:4d}/{epochs} | "
                      f"Loss: {avg:.4f} | Recon: {recon.item():.4f} | KL: {kl.item():.4f}")

        return self

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.encoder.net[0].out_features,
            'loss_factor': self.loss_factor,
            'transformer': self.transformer,
            'train_losses': self.train_losses,
        }, path)

    @classmethod
    def load(cls, path, device=None):
        device = device or DEVICE
        ckpt = torch.load(path, map_location=device)
        model = cls(
            input_dim=ckpt['input_dim'],
            latent_dim=ckpt['latent_dim'],
            hidden_dim=ckpt['hidden_dim'],
            transformer=ckpt['transformer'],
            loss_factor=ckpt['loss_factor'],
        )
        model.load_state_dict(ckpt['state_dict'])
        model.train_losses = ckpt['train_losses']
        return model.to(device)


# ============================================================
# 3. GENERATIVE REPLAY PIPELINE (Per-class TVAEs)
# ============================================================
# 3.5 CORRELATION STRUCTURE COMPARISON VISUALIZATION
# ============================================================

def plot_correlation_comparison(real_df, synthetic_df, feature_columns,
                                title="Correlation Structure", save_path=None,
                                figsize=(18, 5)):
    """
    Plot side-by-side correlation matrices: Real | Synthetic | Difference.
    
    Matches the style from the reference figure:
      - Left:   Real correlation (diverging red-blue)
      - Center: Synthetic correlation (diverging red-blue)
      - Right:  Absolute difference (sequential, lower = better)
    
    Args:
        real_df:          DataFrame with real data
        synthetic_df:     DataFrame with synthetic data
        feature_columns:  list of feature column names to compare
        title:            main title for the figure
        save_path:        if provided, saves figure to this path
        figsize:          figure size tuple
    """
    real_corr = real_df[feature_columns].corr()
    synth_corr = synthetic_df[feature_columns].corr()
    diff_corr = (real_corr - synth_corr).abs()

    # Short labels for readability
    labels = [c[:8] for c in feature_columns]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # Diverging colormap for correlation
    cmap_div = 'RdBu_r'
    # Sequential colormap for difference
    cmap_seq = 'Reds'

    # --- Real ---
    im0 = axes[0].imshow(real_corr.values, cmap=cmap_div, vmin=-1, vmax=1, aspect='equal')
    axes[0].set_title('Real', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # --- Synthetic ---
    im1 = axes[1].imshow(synth_corr.values, cmap=cmap_div, vmin=-1, vmax=1, aspect='equal')
    axes[1].set_title('Synthetic', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=7)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # --- Difference ---
    im2 = axes[2].imshow(diff_corr.values, cmap=cmap_seq, vmin=0, vmax=1, aspect='equal')
    axes[2].set_title('Difference (lower = better)', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_yticklabels(labels, fontsize=7)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")

    plt.show()
    plt.close()

    # Print summary metric
    mean_diff = diff_corr.values[np.triu_indices_from(diff_corr.values, k=1)].mean()
    max_diff = diff_corr.values[np.triu_indices_from(diff_corr.values, k=1)].max()
    print(f"  Mean |corr difference|: {mean_diff:.4f}")
    print(f"  Max  |corr difference|: {max_diff:.4f}")

    return real_corr, synth_corr, diff_corr


def plot_correlation_comparison_per_class(real_df, synthetic_df, feature_columns,
                                          label_column='label', save_path_prefix=None,
                                          figsize=(18, 5)):
    """
    Plot correlation comparison separately for each class.
    Useful to see if per-class TVAE captures within-class structure.
    """
    classes = sorted(real_df[label_column].unique())

    for cls in classes:
        real_cls = real_df[real_df[label_column] == cls]
        synth_cls = synthetic_df[synthetic_df[label_column] == cls]

        if len(synth_cls) < 2:
            print(f"  Skipping class {cls}: too few synthetic samples ({len(synth_cls)})")
            continue

        save_path = None
        if save_path_prefix:
            save_path = f"{save_path_prefix}_class_{cls}.png"

        print(f"\n{'='*60}")
        print(f"Class {cls} — Real: {len(real_cls)} samples, Synthetic: {len(synth_cls)} samples")
        print(f"{'='*60}")

        plot_correlation_comparison(
            real_cls, synth_cls, feature_columns,
            title=f"Correlation Structure — Class {cls}",
            save_path=save_path, figsize=figsize
        )


# ============================================================
# 4. GENERATIVE REPLAY PIPELINE (Per-class TVAEs)
# ============================================================

class GenerativeReplayPipeline:
    """
    Continual learning with generative replay.

    KEY DESIGN: One TVAE per class.
      - Avoids label collapse entirely
      - At replay time, generate N/num_classes samples from each class TVAE
      - Retrain all TVAEs + classifier on combined data

    Flow:
      t0:  Train classifier + per-class TVAEs on D0
      t1:  Generate D0_hat from per-class TVAEs
           Train on D1 ∪ D0_hat, retrain TVAEs
      t2:  Repeat
    """

    def __init__(self,
                 feature_columns,
                 label_column='label',
                 latent_dim=16,
                 hidden_dim=128,
                 tvae_epochs=300,
                 lr=1e-3,
                 batch_size=64,
                 replay_ratio=1.0,
                 n_gmm_components=5):

        self.feature_columns = feature_columns
        self.label_column = label_column
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.tvae_epochs = tvae_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.replay_ratio = replay_ratio
        self.n_gmm_components = n_gmm_components

        # Per-class state
        self.class_tvaes = {}          # label -> TVAE model
        self.class_transformers = {}   # label -> transformer
        self.classifier = None
        self.classes = None
        self.history = []
        self.timestep = 0

    def _get_features(self, df):
        return df[self.feature_columns + [self.label_column]].copy()

    def _train_class_tvaes(self, df, verbose=True):
        """Train one TVAE per class (features only, no label column)."""
        self.classes = sorted(df[self.label_column].unique())

        for cls_label in self.classes:
            cls_data = df[df[self.label_column] == cls_label][self.feature_columns].copy()
            if verbose:
                print(f"\n  -- Class {cls_label} ({len(cls_data)} samples) --")

            # Fit transformer on this class's features
            transformer = TVAEDataTransformer(
                n_gmm_components=self.n_gmm_components,
                discrete_columns=[]   # features only, all continuous
            )
            transformer.fit(cls_data)
            transformed = transformer.transform(cls_data)

            if verbose:
                print(f"     Transformed dim: {transformed.shape[1]}")

            # Train TVAE
            tvae = TVAE(
                input_dim=transformed.shape[1],
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                transformer=transformer,
                loss_factor=2.0,
            )
            tvae.fit(transformed, epochs=self.tvae_epochs,
                     batch_size=self.batch_size, lr=self.lr, verbose=verbose)

            self.class_tvaes[cls_label] = tvae
            self.class_transformers[cls_label] = transformer

    def _generate_replay(self, n_per_class):
        """Generate balanced replay samples from per-class TVAEs."""
        all_dfs = []

        for cls_label in self.classes:
            tvae = self.class_tvaes[cls_label]
            transformer = self.class_transformers[cls_label]

            synthetic_encoded = tvae.generate(n_per_class)
            synthetic_df = transformer.inverse_transform(synthetic_encoded)
            n_nans = synthetic_df.isna().sum().sum()
            if n_nans > 0:
                print(f"    [Warning] Class {cls_label}: {n_nans} NaN values cleaned in synthetic data")
            synthetic_df[self.label_column] = cls_label
            all_dfs.append(synthetic_df)

        return pd.concat(all_dfs, ignore_index=True)

    def initial_train(self, df_train, df_test=None):
        """t0: Initial training."""
        print("=" * 60)
        print(f"TIME STEP t{self.timestep}: Initial Training")
        print("=" * 60)

        data = self._get_features(df_train)
        print(f"Samples: {len(data)} | "
              f"Label dist: {dict(data[self.label_column].value_counts())}")

        # 1. Train per-class TVAEs
        print("\n[1/2] Training per-class TVAEs...")
        self._train_class_tvaes(data)

        # 2. Evaluate: train classifier on real, test on synthetic
        print("\n[2/2] Evaluating (Train on Real, Test on Synthetic)...")
        result = self._evaluate(data, df_test)
        self._quality_check(data)
        self.history.append(result)
        self.timestep += 1
        return self

    def continual_update(self, df_new, df_test=None):
        """t_k (k>0): Generate replay, combine, retrain everything."""
        print("\n" + "=" * 60)
        print(f"TIME STEP t{self.timestep}: Continual Update")
        print("=" * 60)

        new_data = self._get_features(df_new)
        n_new = len(new_data)
        n_per_class = int(n_new * self.replay_ratio / len(self.classes))

        # 1. Generate balanced replay
        print(f"\n[1/3] Generating replay: {n_per_class} samples × "
              f"{len(self.classes)} classes = {n_per_class * len(self.classes)} total")
        replay_df = self._generate_replay(n_per_class)
        print(f"  Replay label dist: {dict(replay_df[self.label_column].value_counts())}")

        # 2. Combine
        combined = pd.concat([new_data, replay_df], ignore_index=True)
        print(f"\n[2/3] Combined: {len(new_data)} real + {len(replay_df)} synthetic "
              f"= {len(combined)} total")
        print(f"  Combined label dist: {dict(combined[self.label_column].value_counts())}")

        # 3. Retrain per-class TVAEs on combined data
        print(f"\n[3/3] Retraining per-class TVAEs...")
        self._train_class_tvaes(combined)

        # 4. Evaluate: train classifier on combined (real+replay), test on fresh synthetic
        print("\n  Evaluating (Train on Real+Replay, Test on Synthetic)...")
        result = self._evaluate(combined, df_test)
        result['n_real'] = n_new
        result['n_synthetic'] = len(replay_df)
        self._quality_check(combined)
        self.history.append(result)
        self.timestep += 1
        return self

    def _evaluate(self, train_data, test_df=None):
        """
        Evaluate with Train on Real, Test on Synthetic (TRTS).
        
        1. Train classifier on real data
        2. Generate synthetic data from per-class TVAEs
        3. Test classifier on synthetic data
        
        High TRTS accuracy = synthetic data faithfully represents real distribution.
        Also tests on held-out real test set if provided.
        """
        X_train = train_data[self.feature_columns].values
        y_train = train_data[self.label_column].values

        # Train classifier on REAL data only
        clf_real = RandomForestClassifier(n_estimators=200, random_state=42)
        clf_real.fit(X_train, y_train)
        self.classifier = clf_real

        train_acc = accuracy_score(y_train, clf_real.predict(X_train))

        result = {
            'timestep': self.timestep,
            'n_real': len(train_data),
            'n_synthetic': 0,
            'train_on_real_acc': train_acc,
        }
        print(f"\n  [Train on Real] Train accuracy: {train_acc:.4f}")

        # Generate synthetic test set
        n_per_class = len(train_data) // max(len(self.classes), 1)
        synthetic_df = self.generate_synthetic(n_per_class)
        X_synth = synthetic_df[self.feature_columns].values
        y_synth = synthetic_df[self.label_column].values

        trts_acc = accuracy_score(y_synth, clf_real.predict(X_synth))
        result['trts_acc'] = trts_acc
        print(f"  [Test on Synthetic] TRTS accuracy: {trts_acc:.4f}")
        print(f"\n  TRTS Classification Report:\n"
              f"{classification_report(y_synth, clf_real.predict(X_synth))}")

        # Also test on held-out real data if provided
        if test_df is not None:
            td = self._get_features(test_df)
            X_test = td[self.feature_columns].values
            y_test = td[self.label_column].values
            test_acc = accuracy_score(y_test, clf_real.predict(X_test))
            result['test_real_acc'] = test_acc
            print(f"  [Test on Real holdout] accuracy: {test_acc:.4f}")

        return result

    def _quality_check(self, real_data):
        """Compare real vs synthetic feature statistics per class."""
        print("  [Quality Check]")
        for cls_label in self.classes:
            real_cls = real_data[real_data[self.label_column] == cls_label]
            tvae = self.class_tvaes[cls_label]
            transformer = self.class_transformers[cls_label]

            synth_enc = tvae.generate(len(real_cls))
            synth_df = transformer.inverse_transform(synth_enc)

            print(f"\n  Class {cls_label}:")
            print(f"  {'Column':<12} {'Real Mean':>12} {'Synth Mean':>12} "
                  f"{'Real Std':>12} {'Synth Std':>12}")
            print(f"  {'-' * 60}")
            for col in self.feature_columns[:5]:
                rm = real_cls[col].mean()
                sm = synth_df[col].mean() if col in synth_df else float('nan')
                rs = real_cls[col].std()
                ss = synth_df[col].std() if col in synth_df else float('nan')
                print(f"  {col:<12} {rm:>12.4f} {sm:>12.4f} {rs:>12.4f} {ss:>12.4f}")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("GENERATIVE REPLAY SUMMARY")
        print("=" * 60)
        print(f"  {'Step':<6} {'Real':<7} {'Synth':<7} "
              f"{'Train(R)':<10} {'TRTS':<10} {'Test(R)':<10}")
        print(f"  {'-'*56}")
        for h in self.history:
            train_r = f"{h.get('train_on_real_acc', 0):.4f}"
            trts = f"{h.get('trts_acc', 0):.4f}" if 'trts_acc' in h else 'N/A'
            test_r = f"{h.get('test_real_acc', 0):.4f}" if 'test_real_acc' in h else 'N/A'
            print(f"  t{h['timestep']:<5} {h['n_real']:<7} {h['n_synthetic']:<7} "
                  f"{train_r:<10} {trts:<10} {test_r:<10}")
        print()
        print("  Train(R) = classifier trained & tested on real data")
        print("  TRTS     = trained on real, tested on synthetic (higher = better generation)")
        print("  Test(R)  = trained on real, tested on held-out real data")

    def generate_synthetic(self, n_per_class=100):
        """Generate a full synthetic DataFrame with labels from per-class TVAEs."""
        all_dfs = []
        for cls_label in self.classes:
            tvae = self.class_tvaes[cls_label]
            transformer = self.class_transformers[cls_label]
            synth_enc = tvae.generate(n_per_class)
            synth_df = transformer.inverse_transform(synth_enc)
            synth_df[self.label_column] = cls_label
            all_dfs.append(synth_df)
        return pd.concat(all_dfs, ignore_index=True)

    def plot_correlation(self, real_df, n_synthetic=None,
                         per_class=True, save_path_prefix=None):
        """
        Generate synthetic data and plot correlation comparison vs real.
        
        Args:
            real_df:           DataFrame with real data (must include label)
            n_synthetic:       samples per class (default: match real per-class counts)
            per_class:         if True, also plot per-class comparisons
            save_path_prefix:  e.g. 'figures/corr' -> saves 'figures/corr_overall.png', etc.
        """
        real_data = self._get_features(real_df)

        # Determine n_per_class
        if n_synthetic is None:
            n_per_class = len(real_data) // len(self.classes)
        else:
            n_per_class = n_synthetic // len(self.classes)

        synthetic_df = self.generate_synthetic(n_per_class)

        # Overall comparison
        save_overall = f"{save_path_prefix}_overall.png" if save_path_prefix else None
        print("\n" + "=" * 60)
        print("OVERALL Correlation Comparison")
        print("=" * 60)
        plot_correlation_comparison(
            real_data, synthetic_df, self.feature_columns,
            title="Correlation Structure — Overall",
            save_path=save_overall
        )

        # Per-class comparison
        if per_class:
            plot_correlation_comparison_per_class(
                real_data, synthetic_df, self.feature_columns,
                label_column=self.label_column,
                save_path_prefix=save_path_prefix
            )
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        for cls_label, tvae in self.class_tvaes.items():
            tvae.save(os.path.join(directory, f'tvae_class_{cls_label}.pt'))
        with open(os.path.join(directory, 'pipeline_state.pkl'), 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'classes': self.classes,
                'feature_columns': self.feature_columns,
                'label_column': self.label_column,
                'history': self.history,
                'timestep': self.timestep,
            }, f)
        print(f"  Pipeline saved to {directory}/")


# ============================================================
# 4. DEMO: Run on uploaded dataset
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print("Loading data...")
    df = pd.read_csv("/Users/souba636/Documents/vinnova_paper_2/Federated_IDS/attack_data/worstparent_var15_base/1_features_timeseries_60_sec.csv")

    # Drop useless columns (all-zero + index)
    drop_cols = ['Unnamed: 0', 'disr', 'diss', 'disr.1', 'diss.1']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    feature_cols = [c for c in df.columns if c != 'label']
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Total samples: {len(df)}\n")

    # ---- Simulate continual learning: 3 time chunks ----
    chunk = len(df) // 3
    df_t0 = df.iloc[:chunk].reset_index(drop=True)
    df_t1 = df.iloc[chunk:2 * chunk].reset_index(drop=True)
    df_t2 = df.iloc[2 * chunk:].reset_index(drop=True)

    global_test = df.sample(n=100, random_state=99)
    print(f"Split: t0={len(df_t0)}, t1={len(df_t1)}, t2={len(df_t2)}, "
          f"global_test={len(global_test)}\n")

    # ---- Run pipeline ----
    pipeline = GenerativeReplayPipeline(
        feature_columns=feature_cols,
        label_column='label',
        latent_dim=12,
        hidden_dim=64,
        tvae_epochs=1000,
        lr=1e-3,
        batch_size=64,
        replay_ratio=1.0,
        n_gmm_components=3,
    )

    pipeline.initial_train(df_t0, df_test=global_test)
    pipeline.continual_update(df_t1, df_test=global_test)
    pipeline.continual_update(df_t2, df_test=global_test)

    # ---- Baseline: naive (no replay) ----
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

    # ---- Skyline: all data at once ----
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

    # ---- Summary ----
    pipeline.print_summary()

    # ---- Correlation Structure Comparison ----
    # Compare real vs synthetic data correlation matrices
    # Overall + per-class, saved as PNG files
    print("\n" + "=" * 60)
    print("CORRELATION STRUCTURE ANALYSIS")
    print("=" * 60)
    pipeline.plot_correlation(
        df,                              # full real dataset
        n_synthetic=len(df),             # generate same number of samples
        per_class=True,
        save_path_prefix="correlation"   # saves correlation_overall.png, correlation_class_0.0.png, etc.
    )

    # Save
    pipeline.save("tvae_pipeline_pytorch")
    print("\nDone!")