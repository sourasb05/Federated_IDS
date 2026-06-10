"""
TabDDPM-based Generative Replay Pipeline for Continual Learning
================================================================
PyTorch implementation — mirrors tvae_generative_replay_pytorch.py structure.

TabDDPM (Kotelnikov et al., 2023, arXiv:2209.15421):
  - Gaussian diffusion on continuous features
  - Multinomial diffusion on categorical features
  - MLP-based denoising network conditioned on timestep t
  - Per-class generators (one DDPM per class label)

Key differences from TVAE:
  - No encoder / latent space — purely generative (forward + reverse diffusion)
  - No KL collapse — diffusion training is much more stable
  - Slower inference (T denoising steps), but much better distribution coverage
  - Categorical columns handled with multinomial diffusion (not one-hot argmax)

Flow:
  t0:  Train per-class DDPMs on D0
  t1:  Generate D0_hat from frozen DDPMs
       Train on D1 ∪ D0_hat, retrain DDPMs on combined
  t2:  Repeat

References:
  Kotelnikov et al. (2023) TabDDPM: Modelling Tabular Data with Diffusion Models
    arXiv:2209.15421
  Ho et al. (2020) Denoising Diffusion Probabilistic Models
    arXiv:2006.11239
  Austin et al. (2021) Structured Denoising Diffusion Models in Discrete State Spaces
    arXiv:2107.03006  (multinomial diffusion)
"""

import os
import math
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import ks_2samp, wasserstein_distance, gaussian_kde
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# 1. NOISE SCHEDULE
# ============================================================

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule (Nichol & Dhariwal, 2021).
    Produces smaller betas near t=0 and t=T than linear schedule,
    preventing the signal from being destroyed too early.

    Returns betas of shape (T,).
    """
    steps = T + 1
    x     = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-4, max=0.999)


class GaussianDiffusion:
    """
    Gaussian forward process and reverse diffusion for continuous features.

    Forward:  q(x_t | x_0) = N(x_t; sqrt(ᾱ_t)*x_0, (1-ᾱ_t)*I)
    Reverse:  p_θ(x_{t-1}|x_t) — ε_θ predicts the noise added at step t

    All tensors live on `device`.
    """

    def __init__(self, T: int = 1000, device: str = 'cpu'):
        self.T      = T
        self.device = device

        betas               = cosine_beta_schedule(T).to(device)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas               = betas
        self.alphas_cumprod      = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod         = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
        # posterior variance q(x_{t-1}|x_t,x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).clamp(min=1e-20)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Add noise to x0 at diffusion step t → x_t."""
        if noise is None:
            noise = torch.randn_like(x0)
        a  = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sa = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return a * x0 + sa * noise

    def p_mean_variance(self, model, x_t: torch.Tensor,
                        t: torch.Tensor) -> tuple:
        """Compute p_θ(x_{t-1}|x_t) mean and variance."""
        eps_pred = model(x_t, t)
        a  = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sa = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        # reconstruct x0 from predicted noise
        x0_pred = (x_t - sa * eps_pred) / a.clamp(min=1e-8)
        x0_pred = x0_pred.clamp(-3.0, 3.0)

        b  = self.betas[t].view(-1, 1)
        acp= self.alphas_cumprod[t].view(-1, 1)
        acp_prev = self.alphas_cumprod_prev[t].view(-1, 1)

        mean = (
            b * acp_prev.sqrt() / (1.0 - acp) * x0_pred
            + (1.0 - acp_prev) * (1.0 - b).sqrt() / (1.0 - acp) * x_t
        )
        var = self.posterior_variance[t].view(-1, 1)
        return mean, var

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor,
                 t_scalar: int) -> torch.Tensor:
        """One reverse denoising step."""
        B   = x_t.size(0)
        t   = torch.full((B,), t_scalar, device=self.device, dtype=torch.long)
        mean, var = self.p_mean_variance(model, x_t, t)
        noise = torch.randn_like(x_t) if t_scalar > 0 else torch.zeros_like(x_t)
        return mean + var.sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n: int, feature_dim: int,
               ddim_steps: int = None) -> torch.Tensor:
        """
        Full reverse diffusion: x_T ~ N(0,I) → x_0.

        ddim_steps: if set, use a subset of T steps (faster inference).
        Returns (n, feature_dim) tensor in approximately the data range.
        """
        model.eval()
        x = torch.randn(n, feature_dim, device=self.device)

        steps = list(reversed(range(self.T)))
        if ddim_steps is not None and ddim_steps < self.T:
            # evenly spaced subset
            idx   = np.linspace(0, self.T - 1, ddim_steps, dtype=int)
            steps = sorted(idx.tolist(), reverse=True)

        for t_scalar in steps:
            x = self.p_sample(model, x, t_scalar)

        model.train()
        return x


# ============================================================
# 2. DENOISING NETWORK (MLP with sinusoidal time embedding)
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timestep t.
    Maps integer t → dense embedding of size `dim`.
    Same approach as transformer positional encoding (Vaswani et al. 2017).
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) long
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)    # (B, dim)
        return emb


class ResidualBlock(nn.Module):
    """MLP residual block with time-step conditioning."""

    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2   = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.linear1(h) + self.time_proj(t_emb)
        h = self.dropout(self.act(self.norm2(h)))
        h = self.linear2(h)
        return x + h


class MLPDenoiser(nn.Module):
    """
    MLP denoising network ε_θ(x_t, t) for Gaussian diffusion.

    Architecture:
      x_t (B, D) → input projection
      t   (B,)   → sinusoidal embedding → time projection
      N residual blocks with time conditioning
      → output projection → ε̂ (B, D)

    Predicts the noise ε that was added at step t to x_0,
    i.e. minimises E[||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²].
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256,
                 n_blocks: int = 4, time_emb_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.time_emb   = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp   = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.blocks     = nn.ModuleList([
            ResidualBlock(hidden_dim, time_emb_dim, dropout)
            for _ in range(n_blocks)
        ])
        self.out_norm   = nn.LayerNorm(hidden_dim)
        self.out_proj   = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, D)  noisy input at step t
        t : (B,)    integer diffusion timesteps
        returns: ε̂ (B, D) predicted noise
        """
        t_emb = self.time_mlp(self.time_emb(t))   # (B, time_emb_dim)
        h     = self.input_proj(x)                  # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(self.out_norm(h))       # (B, D)


# ============================================================
# 3. TABDDPM MODEL  (continuous features only)
# ============================================================

class TabDDPM:
    """
    TabDDPM for continuous tabular features.

    Wraps GaussianDiffusion + MLPDenoiser into a sklearn-style interface
    that mirrors the TVAE.fit() / .generate() API used in the pipeline.

    Continuous features are standardised (zero-mean, unit-variance) before
    diffusion and de-standardised at generation time, matching the
    original TabDDPM paper's pre-processing approach.
    """

    def __init__(self,
                 feature_dim  : int,
                 T            : int   = 1000,
                 hidden_dim   : int   = 256,
                 n_blocks     : int   = 4,
                 time_emb_dim : int   = 128,
                 dropout      : float = 0.1,
                 lr           : float = 1e-3,
                 batch_size   : int   = 512,
                 epochs       : int   = 500,
                 ddim_steps   : int   = 200,
                 device       : str   = 'cpu'):

        self.feature_dim  = feature_dim
        self.T            = T
        self.hidden_dim   = hidden_dim
        self.n_blocks     = n_blocks
        self.time_emb_dim = time_emb_dim
        self.dropout      = dropout
        self.lr           = lr
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.ddim_steps   = ddim_steps
        self.device       = device

        self.diffusion = GaussianDiffusion(T=T, device=device)
        self.model     = MLPDenoiser(feature_dim, hidden_dim,
                                     n_blocks, time_emb_dim, dropout).to(device)
        self.train_losses: list = []

        # standardisation statistics (fit on training data)
        self._mean: np.ndarray = None
        self._std:  np.ndarray = None

    # ── internal helpers ─────────────────────────────────────

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / (self._std + 1e-8)

    def _destandardise(self, X: np.ndarray) -> np.ndarray:
        return X * (self._std + 1e-8) + self._mean

    # ── public API ───────────────────────────────────────────

    def fit(self, X_np: np.ndarray, verbose: bool = True) -> 'TabDDPM':
        """
        Train the denoising network on continuous feature matrix X_np.

        X_np : (N, D)  float32
        """
        # standardise
        self._mean = X_np.mean(axis=0).astype(np.float32)
        self._std  = X_np.std(axis=0).astype(np.float32)
        X_std = self._standardise(X_np).astype(np.float32)

        dataset   = TensorDataset(torch.tensor(X_std, device=self.device))
        loader    = DataLoader(dataset, batch_size=self.batch_size,
                               shuffle=True, drop_last=False)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.1
        )

        self.model.train()
        self.train_losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches  = 0

            for (x0,) in loader:
                # sample random diffusion timesteps
                t     = torch.randint(0, self.T, (x0.size(0),),
                                      device=self.device)
                noise = torch.randn_like(x0)
                x_t   = self.diffusion.q_sample(x0, t, noise)

                # predict noise, compute MSE loss
                eps_pred = self.model(x_t, t)
                loss     = F.mse_loss(eps_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"    Epoch {epoch+1:4d}/{self.epochs} | "
                      f"Loss: {avg:.6f}")

        return self

    @torch.no_grad()
    def generate(self, n: int) -> np.ndarray:
        """
        Sample n synthetic rows.
        Returns (n, D) float32 numpy array in original feature space.
        """
        X_std = self.diffusion.sample(
            self.model, n, self.feature_dim,
            ddim_steps=self.ddim_steps
        )
        X_np = X_std.cpu().numpy().astype(np.float32)
        return self._destandardise(X_np)

    def save(self, path: str):
        torch.save({
            'model_state'  : self.model.state_dict(),
            'feature_dim'  : self.feature_dim,
            'T'            : self.T,
            'hidden_dim'   : self.hidden_dim,
            'n_blocks'     : self.n_blocks,
            'time_emb_dim' : self.time_emb_dim,
            'dropout'      : self.dropout,
            'lr'           : self.lr,
            'batch_size'   : self.batch_size,
            'epochs'       : self.epochs,
            'ddim_steps'   : self.ddim_steps,
            'device'       : self.device,
            'train_losses' : self.train_losses,
            'mean'         : self._mean,
            'std'          : self._std,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = None) -> 'TabDDPM':
        ckpt   = torch.load(path, map_location=device or 'cpu')
        obj    = cls(
            feature_dim  = ckpt['feature_dim'],
            T            = ckpt['T'],
            hidden_dim   = ckpt['hidden_dim'],
            n_blocks     = ckpt['n_blocks'],
            time_emb_dim = ckpt['time_emb_dim'],
            dropout      = ckpt['dropout'],
            lr           = ckpt['lr'],
            batch_size   = ckpt['batch_size'],
            epochs       = ckpt['epochs'],
            ddim_steps   = ckpt['ddim_steps'],
            device       = device or ckpt['device'],
        )
        obj.model.load_state_dict(ckpt['model_state'])
        obj.train_losses = ckpt['train_losses']
        obj._mean        = ckpt['mean']
        obj._std         = ckpt['std']
        return obj


# ============================================================
# 4. CORRELATION STRUCTURE VISUALISATION  (same as TVAE file)
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
            title=f"Correlation Structure — Class {cls}",
            save_path=save_path, figsize=figsize
        )


# ============================================================
# 5. GENERATIVE REPLAY PIPELINE  (Per-class TabDDPMs)
# ============================================================

class TabDDPMGenerativeReplayPipeline:
    """
    Continual learning with TabDDPM generative replay.

    Mirrors GenerativeReplayPipeline from tvae_generative_replay_pytorch.py.

    KEY DESIGN: One TabDDPM per class.
      - Each class has its own denoising network trained on that class's features
      - At replay: generate proportional samples from each class DDPM
      - Retrains all DDPMs on combined (real + replay) data each timestep

    Why per-class?
      - Avoids label collapse: the generator never has to learn a
        bimodal distribution and guess which mode to sample from.
      - Class proportions can be controlled exactly at generation time.
    """

    def __init__(self,
                 feature_columns : list,
                 label_column    : str   = 'label',
                 T               : int   = 1000,
                 hidden_dim      : int   = 256,
                 n_blocks        : int   = 4,
                 time_emb_dim    : int   = 128,
                 dropout         : float = 0.0,
                 ddpm_epochs     : int   = 500,
                 ddim_steps      : int   = 200,
                 lr              : float = 1e-3,
                 batch_size      : int   = 512,
                 replay_ratio    : float = 1.0,
                 device          : str   = 'cpu'):

        self.feature_columns = feature_columns
        self.label_column    = label_column
        self.T               = T
        self.hidden_dim      = hidden_dim
        self.n_blocks        = n_blocks
        self.time_emb_dim    = time_emb_dim
        self.dropout         = dropout
        self.ddpm_epochs     = ddpm_epochs
        self.ddim_steps      = ddim_steps
        self.lr              = lr
        self.batch_size      = batch_size
        self.replay_ratio    = replay_ratio
        self.device          = device

        self.class_ddpms: dict = {}     # label -> TabDDPM
        self.classifier         = None
        self.classes            = None
        self.history: list      = []
        self.timestep: int      = 0

    # ── internal helpers ─────────────────────────────────────

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.feature_columns + [self.label_column]].copy()

    def _train_class_ddpms(self, df: pd.DataFrame, verbose: bool = True):
        """Train one TabDDPM per class (features only, no label column)."""
        self.classes = sorted(df[self.label_column].unique())

        for cls_label in self.classes:
            cls_data = df[df[self.label_column] == cls_label][self.feature_columns]
            X_np = cls_data.values.astype(np.float32)

            if verbose:
                print(f"\n  -- Class {cls_label} ({len(X_np)} samples) --")

            ddpm = TabDDPM(
                feature_dim  = X_np.shape[1],
                T            = self.T,
                hidden_dim   = self.hidden_dim,
                n_blocks     = self.n_blocks,
                time_emb_dim = self.time_emb_dim,
                dropout      = self.dropout,
                lr           = self.lr,
                batch_size   = self.batch_size,
                epochs       = self.ddpm_epochs,
                ddim_steps   = self.ddim_steps,
                device       = self.device,
            )
            ddpm.fit(X_np, verbose=verbose)
            self.class_ddpms[cls_label] = ddpm

    def _generate_replay(self, n_per_class: int) -> pd.DataFrame:
        """Generate balanced replay samples from per-class DDPMs."""
        all_dfs = []
        for cls_label in self.classes:
            ddpm  = self.class_ddpms[cls_label]
            X_syn = ddpm.generate(n_per_class)
            df_syn = pd.DataFrame(X_syn, columns=self.feature_columns)
            df_syn[self.label_column] = cls_label
            all_dfs.append(df_syn)
        return pd.concat(all_dfs, ignore_index=True)

    def _quality_check(self, real_data: pd.DataFrame):
        """Compare real vs synthetic feature statistics per class."""
        print("  [Quality Check]")
        for cls_label in self.classes:
            real_cls = real_data[real_data[self.label_column] == cls_label]
            ddpm     = self.class_ddpms[cls_label]
            X_syn    = ddpm.generate(len(real_cls))
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
            'timestep'        : self.timestep,
            'n_real'          : len(train_data),
            'n_synthetic'     : 0,
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
                      df_test: pd.DataFrame = None) -> 'TabDDPMGenerativeReplayPipeline':
        """t0: Initial training."""
        print("=" * 60)
        print(f"TIME STEP t{self.timestep}: Initial Training")
        print("=" * 60)

        data = self._get_features(df_train)
        print(f"Samples: {len(data)} | "
              f"Label dist: {dict(data[self.label_column].value_counts())}")

        print("\n[1/2] Training per-class TabDDPMs...")
        self._train_class_ddpms(data)

        print("\n[2/2] Evaluating (TRTS)...")
        result = self._evaluate(data, df_test)
        self._quality_check(data)
        self.history.append(result)
        self.timestep += 1
        return self

    def continual_update(self, df_new: pd.DataFrame,
                         df_test: pd.DataFrame = None) -> 'TabDDPMGenerativeReplayPipeline':
        """t_k (k>0): Generate replay, combine, retrain DDPMs."""
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

        print(f"\n[3/3] Retraining per-class TabDDPMs on combined data...")
        self._train_class_ddpms(combined)

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
            X_syn  = self.class_ddpms[cls_label].generate(n_per_class)
            df_syn = pd.DataFrame(X_syn, columns=self.feature_columns)
            df_syn[self.label_column] = cls_label
            all_dfs.append(df_syn)
        return pd.concat(all_dfs, ignore_index=True)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("TABDDPM GENERATIVE REPLAY SUMMARY")
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
        real_data   = self._get_features(real_df)
        n_per_class = (len(real_data) // len(self.classes)
                       if n_synthetic is None else n_synthetic // len(self.classes))
        synthetic_df = self.generate_synthetic(n_per_class)

        save_overall = f"{save_path_prefix}_overall.png" if save_path_prefix else None
        print("\n" + "=" * 60)
        print("OVERALL Correlation Comparison")
        print("=" * 60)
        plot_correlation_comparison(
            real_data, synthetic_df, self.feature_columns,
            title="Correlation Structure — Overall (TabDDPM)",
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
                                   save_dir      : str  = 'results',
                                   plots_per_row : int  = 5) -> pd.DataFrame:
        """
        Compare real vs synthetic distributions with:
          1. KS test (per feature, per class)          — measures distributional similarity
          2. Wasserstein distance (per feature)        — earth-mover's distance
          3. Distribution overlay plots (KDE + hist)   — visual per-feature comparison
          4. KS summary bar chart                      — p-values across all features

        Parameters
        ----------
        real_df      : full real DataFrame (must contain label column)
        n_synthetic  : samples to generate per class (default: match real per-class count)
        save_dir     : directory for saved PNGs
        plots_per_row: feature subplots per row in distribution grid

        Returns
        -------
        ks_summary : pd.DataFrame with columns
                     [class, feature, ks_stat, p_value, wasserstein, similar]
                     sorted by ks_stat descending (worst first)
        """
        os.makedirs(save_dir, exist_ok=True)
        real_data = self._get_features(real_df)

        n_classes = len(self.classes) if self.classes else 1
        if n_synthetic is None:
            n_per_class = len(real_data) // max(n_classes, 1)
        else:
            n_per_class = n_synthetic // max(n_classes, 1)

        all_rows = []

        for cls_label in self.classes:
            real_cls = real_data[real_data[self.label_column] == cls_label][self.feature_columns]
            X_syn    = self.class_ddpms[cls_label].generate(n_per_class)
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
                flag = 'OK  ' if similar else 'WARN'
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
            # One subplot per feature, sorted worst KS first
            feat_rows_sorted = sorted(feat_rows,
                                      key=lambda r: r['ks_stat'], reverse=True)
            n_feat    = len(feat_rows_sorted)
            ncols     = min(plots_per_row, n_feat)
            nrows     = math.ceil(n_feat / ncols) if ncols > 0 else 1

            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(4.0 * ncols, 3.2 * nrows),
                                     constrained_layout=True)
            axes_flat = np.array(axes).flatten() if n_feat > 1 else [axes]

            for ax, row in zip(axes_flat, feat_rows_sorted):
                col = row['feature']
                r   = real_cls[col].dropna().values.astype(float)
                s   = syn_cls[col].dropna().values.astype(float)

                # histogram
                ax.hist(r, bins=30, alpha=0.35, color='steelblue',
                        density=True, label='Real')
                ax.hist(s, bins=30, alpha=0.35, color='darkorange',
                        density=True, label='Synthetic')

                # KDE
                try:
                    xs = np.linspace(min(r.min(), s.min()),
                                     max(r.max(), s.max()), 300)
                    ax.plot(xs, gaussian_kde(r)(xs), color='steelblue',  lw=1.5)
                    ax.plot(xs, gaussian_kde(s)(xs), color='darkorange', lw=1.5)
                except Exception:
                    pass

                status = 'OK' if row['similar'] else 'WARN'
                ax.set_title(
                    f"{col}\nKS={row['ks_stat']:.3f}  "
                    f"p={row['p_value']:.3f}  [{status}]",
                    fontsize=7
                )
                ax.legend(fontsize=6)
                ax.tick_params(labelsize=5)
                ax.set_xlabel('value',   fontsize=6)
                ax.set_ylabel('density', fontsize=6)

            # hide unused axes
            for ax in axes_flat[n_feat:]:
                ax.set_visible(False)

            n_sim = sum(1 for r in feat_rows if r['similar'])
            fig.suptitle(
                f"Real vs Synthetic — Class {cls_label}  "
                f"({n_sim}/{n_feat} features pass KS p>0.05)\n"
                f"TabDDPM | sorted worst → best KS (top-left = hardest)",
                fontsize=10
            )
            dist_path = os.path.join(
                save_dir, f'tabddpm_dist_class{cls_label}.png'
            )
            plt.savefig(dist_path, dpi=110, bbox_inches='tight')
            plt.close(fig)
            print(f"  Distribution plot saved → {dist_path}")

            # ── KS p-value bar chart ─────────────────────────────
            # features sorted ascending by p-value (hardest first on left)
            feat_rows_asc = sorted(feat_rows, key=lambda r: r['p_value'])
            feat_names    = [r['feature'] for r in feat_rows_asc]
            p_values      = [r['p_value'] for r in feat_rows_asc]
            ks_stats      = [r['ks_stat'] for r in feat_rows_asc]

            fig2, (ax1, ax2) = plt.subplots(2, 1,
                                             figsize=(max(10, n_feat * 0.35), 8),
                                             constrained_layout=True)

            # p-value bar chart
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
            ax1.set_xticklabels(feat_names, rotation=45,
                                ha='right', fontsize=7)
            ax1.legend(fontsize=8)
            ax1.set_ylim(0, max(max(p_values) * 1.15, 0.1))

            # KS statistic bar chart
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
            ax2.set_xticklabels(feat_names, rotation=45,
                                ha='right', fontsize=7)
            ax2.legend(fontsize=8)

            fig2.suptitle(
                f'KS Test Summary — TabDDPM — Class {cls_label}',
                fontsize=12, fontweight='bold'
            )
            ks_path = os.path.join(
                save_dir, f'tabddpm_ks_class{cls_label}.png'
            )
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

        # save CSV
        csv_path = os.path.join(save_dir, 'tabddpm_ks_results.csv')
        ks_summary.to_csv(csv_path, index=False)
        print(f"  KS results CSV    → {csv_path}")

        return ks_summary

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for cls_label, ddpm in self.class_ddpms.items():
            ddpm.save(os.path.join(directory, f'ddpm_class_{cls_label}.pt'))
        with open(os.path.join(directory, 'pipeline_state.pkl'), 'wb') as f:
            pickle.dump({
                'classifier'     : self.classifier,
                'classes'        : self.classes,
                'feature_columns': self.feature_columns,
                'label_column'   : self.label_column,
                'history'        : self.history,
                'timestep'       : self.timestep,
            }, f)
        print(f"  Pipeline saved to {directory}/")


# ============================================================
# 6. DEMO
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
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

    # ── Simulate continual learning: 3 time chunks ───────────
    chunk  = len(df) // 3
    df_t0  = df.iloc[:chunk].reset_index(drop=True)
    df_t1  = df.iloc[chunk:2 * chunk].reset_index(drop=True)
    df_t2  = df.iloc[2 * chunk:].reset_index(drop=True)

    global_test = df.sample(n=100, random_state=99)
    print(f"Split: t0={len(df_t0)}, t1={len(df_t1)}, t2={len(df_t2)}, "
          f"global_test={len(global_test)}\n")

    # ── Run TabDDPM pipeline ──────────────────────────────────
    pipeline = TabDDPMGenerativeReplayPipeline(
        feature_columns = feature_cols,
        label_column    = 'label',
        T               = 1000,       # diffusion timesteps
        hidden_dim      = 256,        # MLP width
        n_blocks        = 4,          # residual blocks
        time_emb_dim    = 128,        # sinusoidal embedding dim
        dropout         = 0.3,
        ddpm_epochs     = 1000,        # training epochs per DDPM
        ddim_steps      = 200,        # inference steps (< T → faster)
        lr              = 1e-3,
        batch_size      = 512,
        replay_ratio    = 1.0,
        device          = str(DEVICE),
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
    print("\n" + "=" * 60)
    print("DISTRIBUTION COMPARISON & KS STATISTICAL TESTS")
    print("=" * 60)
    ks_results = pipeline.evaluate_synthetic_quality(
        real_df     = df,
        n_synthetic = len(df),      # generate as many synthetic as real
        save_dir    = 'results',    # PNGs + CSV saved here
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
        df,
        n_synthetic        = len(df),
        per_class          = True,
        save_path_prefix   = "results/tabddpm_correlation"
    )

    # ── Save pipeline ─────────────────────────────────────────
    pipeline.save("tabddpm_pipeline_pytorch")
    print("\nDone!")
