# blackhole_vae.py
#
# Specialised synthetic-data generator for Blackhole attacks.
#
# Why a separate strategy is needed
# ───────────────────────────────────
# Blackhole is the hardest attack family to synthesise because:
#   1. disr / diss are structurally ZERO in BOTH normal and attack — dead features.
#   2. The means of dior/dios/tots barely shift (≈ 0.43 vs 0.42 under attack).
#   3. The attack signal lives entirely in the TAIL: normal skew ≈ 1.3, attack skew ≈ 20.
#   4. rank/rank.1 carry a real mean shift (~100 units) but are continuous.
#   5. Standard MSE loss collapses both classes to the same mean, missing the tail.
#
# Strategy implemented here
# ─────────────────────────
# Stage 1 — Background VAE
#   Trained on ALL rows (label 0 + label 1) to learn the shared background
#   distribution of normal RPL traffic. Uses lognormal NLL loss (predicts
#   μ and σ explicitly) for right-skewed features, so it actually captures
#   the heavy tail rather than just the mean.
#
# Stage 2 — Attack-residual model
#   Computes the "delta" between each attack window and its nearest-neighbour
#   background reconstruction.  Trains a second small VAE on those deltas only.
#   At sample time: sample background → add attack delta → clamp to valid range.
#   This forces the model to learn ONLY what distinguishes attack from normal,
#   bypassing the dominant background variation that confuses a single VAE.
#
# Mixture decoder on dior/dios/tots/rank
#   For features whose attack distribution is a mixture of a "normal-looking"
#   bulk and a rare extreme tail, the decoder outputs THREE heads per feature:
#     - background component: (μ_bg, σ_bg) fixed from training data statistics
#     - tail component:        (μ_tail, σ_tail) free parameters learned by VAE
#     - mixing weight:          π (gate) — scalar per feature per timestep
#   Reconstruction loss = -log[ π·N(x; μ_tail, σ_tail) + (1-π)·N(x; μ_bg, σ_bg) ]
#
# Extreme-quantile conditioning
#   Before training the attack VAE, each attack window is tagged is_extreme=1
#   if its mean tots > 90th percentile of the normal distribution.
#   This binary flag is appended as an extra feature during training and sampling,
#   so the VAE learns TWO sub-modes: "normal-looking attack" vs "burst attack".
#
# Evaluation
#   In addition to standard TSTR/TRTS and KS, we report:
#     - Tail KS:  KS computed only on samples above the 90th real-normal percentile
#     - Tail coverage: fraction of real extreme-attack windows whose feature values
#       fall within [min, max] of synthetic extreme-attack windows
#     - TRTS conditional precision: among synthetic attack windows classified as
#       attack by a REAL-data-trained classifier, what fraction are correct?
#
# Run:
#   conda run -n vinnova python toy/blackhole_vae.py
#
# Output:
#   toy/blackhole_vae_results/<variant>/  — per-variant results
#   toy/blackhole_vae_results/summary.json

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
from rvae import REncoder

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'attack_data')
OUT_ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'blackhole_vae_results')
os.makedirs(OUT_ROOT, exist_ok=True)

# Run on all 12 blackhole variants
VARIANTS = [d for d in sorted(os.listdir(DATA_ROOT)) if d.startswith('blackhole_')]

WINDOW_SIZE        = 10
N_SYNTH_PER_CLASS  = 1000
BG_EPOCHS          = 300      # Stage-1 background VAE (600 was overkill for this model size)
DELTA_EPOCHS       = 300      # Stage-2 residual VAE
DEVICE             = 'cpu'    # MPS is slower than CPU for this small GRU (dispatch overhead > compute gain)
RANK_COLS          = ['rank', 'rank.1']
SPARSE_THRESH      = 0.30
TAIL_QUANTILE      = 0.90     # 90th pct of normal tots used to flag extreme windows
LOGNORM_SKEW_THRESH = 1.0

# Features whose attack signal is in the tail → get mixture decoder treatment
MIXTURE_FEATS = ['dior', 'dios', 'tots', 'dior.1', 'dios.1', 'tots.1', 'rank', 'rank.1']

# Seed for reproducible train/test split
RNG_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# (mirrors zirvae_multifile.py exactly so results are comparable)
# ─────────────────────────────────────────────────────────────────────────────

def load_folder(folder_path: str) -> pd.DataFrame:
    dfs = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(folder_path, fname),
                         encoding='utf-8', encoding_errors='ignore')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def split_files(folder_path: str):
    """Return (train_df, test_df) using same 14/6 random split as zirvae_multifile."""
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    rng       = np.random.default_rng(seed=RNG_SEED)
    shuffled  = rng.permutation(all_files).tolist()
    train_f, test_f = shuffled[:14], shuffled[14:]

    def _load(files):
        dfs = []
        for fname in files:
            df = pd.read_csv(os.path.join(folder_path, fname),
                             encoding='utf-8', encoding_errors='ignore')
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    return _load(train_f), _load(test_f)


def preprocess_and_window(df_tr: pd.DataFrame, df_te: pd.DataFrame,
                          window_size: int):
    """
    Feature engineering (fitted on train only):
      1. Differencing + sign-log1p on rank columns
      2. log1p on sparse columns (>SPARSE_THRESH zeros)
      3. Min-max normalisation → [0, 1]
      4. Symmetric rescaling for rank columns (0 → 0.5)
    Returns X_tr, y_tr, X_te, y_te, feat_cols, preproc_params
    """
    feat_cols = [c for c in df_tr.columns if c != 'label']
    rank_present = [c for c in RANK_COLS if c in feat_cols]

    for df in [df_tr, df_te]:
        if rank_present:
            diff = df[rank_present].diff().fillna(0)
            df[rank_present] = np.sign(diff) * np.log1p(np.abs(diff))

    sparse_cols = [c for c in feat_cols
                   if (df_tr[c] == 0).mean() > SPARSE_THRESH]
    for df in [df_tr, df_te]:
        if sparse_cols:
            df[sparse_cols] = np.log1p(df[sparse_cols])

    g_min  = df_tr[feat_cols].min()
    g_max  = df_tr[feat_cols].max()
    denom  = (g_max - g_min).replace(0, 1)
    for df in [df_tr, df_te]:
        df[feat_cols] = ((df[feat_cols] - g_min) / denom).clip(0, 1).fillna(0)

    for col in rank_present:
        abs_max = df_tr[col].abs().max()
        if abs_max > 0:
            for df in [df_tr, df_te]:
                df[col] = (df[col] / (2 * abs_max) + 0.5).clip(0, 1)

    preproc = {'g_min': g_min, 'g_max': g_max, 'denom': denom,
               'sparse_cols': sparse_cols, 'rank_present': rank_present,
               'feat_cols': feat_cols}

    def _windows(df):
        vals   = df[feat_cols].values.astype(np.float32)
        labels = df['label'].values.astype(int)
        X, y   = [], []
        for i in range(len(vals) - window_size):
            lbl = 1 if 1 in labels[i:i + window_size] else 0
            X.append(vals[i:i + window_size])
            y.append(lbl)
        return np.array(X, np.float32), np.array(y, np.int64)

    X_tr, y_tr = _windows(df_tr)
    X_te, y_te = _windows(df_te)
    return X_tr, y_tr, X_te, y_te, feat_cols, preproc


# ─────────────────────────────────────────────────────────────────────────────
# EXTREME-QUANTILE FLAG
# ─────────────────────────────────────────────────────────────────────────────

def add_extreme_flag(X: np.ndarray, y: np.ndarray,
                     feat_cols: list, tail_thresh: float) -> tuple:
    """
    Append a binary is_extreme feature (last column).
    is_extreme = 1 if mean(tots) across the window > tail_thresh.
    tail_thresh is computed from the normal-class training windows.
    Returns (X_aug, feat_cols_aug, tail_thresh_used).
    """
    tots_idx = feat_cols.index('tots') if 'tots' in feat_cols else None
    if tots_idx is None:
        # No tots feature — use first available mixture feature
        for f in MIXTURE_FEATS:
            if f in feat_cols:
                tots_idx = feat_cols.index(f)
                break
    if tots_idx is None:
        dummy = np.zeros((len(X), X.shape[1], 1), dtype=np.float32)
        return np.concatenate([X, dummy], axis=2), feat_cols + ['is_extreme'], 0.0

    normal_tots = X[y == 0, :, tots_idx].mean(axis=1)   # mean over window
    thresh      = float(np.quantile(normal_tots, tail_thresh))
    window_tots = X[:, :, tots_idx].mean(axis=1)
    flag        = (window_tots > thresh).astype(np.float32).reshape(-1, 1, 1)
    flag_tiled  = np.tile(flag, (1, X.shape[1], 1))     # (N, T, 1)
    X_aug       = np.concatenate([X, flag_tiled], axis=2)
    return X_aug, feat_cols + ['is_extreme'], thresh


# ─────────────────────────────────────────────────────────────────────────────
# LOGNORMAL NLL LOSS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def lognormal_nll(x: torch.Tensor,
                  log_mu: torch.Tensor,
                  log_sigma: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of log-normal distribution.
    x         : target values in [0, 1] (already log1p + normalised)
    log_mu    : predicted log-space mean  (unconstrained)
    log_sigma : predicted log-space std   (we softplus to keep > 0)
    We treat x as already in log-space (log1p was applied), so model as Gaussian
    in log-space is equivalent to log-normal in original space.
    """
    sigma = F.softplus(log_sigma) + 1e-4
    nll   = 0.5 * ((x - log_mu) / sigma).pow(2) + torch.log(sigma)
    return nll.mean()


def mixture_nll(x: torch.Tensor,
                log_mu_tail: torch.Tensor,
                log_sigma_tail: torch.Tensor,
                mu_bg: float, sigma_bg: float,
                log_pi: torch.Tensor) -> torch.Tensor:
    """
    2-component mixture NLL for one feature:
      - Background component: N(mu_bg, sigma_bg) — fixed from training data stats
      - Tail component:       N(log_mu_tail, softplus(log_sigma_tail)) — free
      - Mixing weight:        sigmoid(log_pi) for tail, 1-sigmoid for background
    """
    sigma_tail = F.softplus(log_sigma_tail) + 1e-4
    pi_tail    = torch.sigmoid(log_pi)          # weight of tail component

    # log-prob under each component
    def log_normal_pdf(val, mu, sigma):
        return -0.5 * ((val - mu) / sigma).pow(2) - torch.log(sigma) - 0.9189

    lp_tail = log_normal_pdf(x, log_mu_tail, sigma_tail)
    lp_bg   = log_normal_pdf(x,
                              torch.full_like(x, mu_bg),
                              torch.full_like(x, sigma_bg))

    # log-sum-exp for numerical stability
    log_mix  = torch.logaddexp(
        torch.log(pi_tail    + 1e-9) + lp_tail,
        torch.log(1 - pi_tail + 1e-9) + lp_bg
    )
    return -log_mix.mean()


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND VAE  (Stage 1)
# Trained on ALL windows (normal + attack) to learn the shared background.
# Mixture decoder for MIXTURE_FEATS, BCE for Bernoulli, MSE for rest.
# ─────────────────────────────────────────────────────────────────────────────

class MixtureDecoder(nn.Module):
    """
    GRU decoder with per-feature specialised heads:
      bernoulli_idx  → raw logit head (BCE loss)
      mixture_idx    → (mu_tail, log_sigma_tail, log_pi) — mixture NLL loss
      zi_idx         → gate logit + value head (ZI-lognormal)
      continuous_idx → sigmoid output + MSE
    """
    def __init__(self, latent_dim: int, hidden_dim: int,
                 n_features: int, window_size: int,
                 bernoulli_idx: list, mixture_idx: list,
                 zi_idx: list, continuous_idx: list,
                 n_layers: int = 1):
        super().__init__()
        self.window_size    = window_size
        self.n_layers       = n_layers
        self.hidden_dim     = hidden_dim
        self.n_features     = n_features
        self.bernoulli_idx  = bernoulli_idx
        self.mixture_idx    = mixture_idx
        self.zi_idx         = zi_idx
        self.continuous_idx = continuous_idx

        self.fc_h0    = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_input = nn.Linear(latent_dim, n_features)
        self.gru      = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True)

        # Shared → n_features (for Bernoulli/ZI/continuous heads)
        self.fc_shared = nn.Linear(hidden_dim, n_features)

        # Mixture heads: 3 outputs per mixture feature (mu_tail, log_sigma, log_pi)
        if mixture_idx:
            self.fc_mix_mu    = nn.Linear(hidden_dim, len(mixture_idx))
            self.fc_mix_sigma = nn.Linear(hidden_dim, len(mixture_idx))
            self.fc_mix_pi    = nn.Linear(hidden_dim, len(mixture_idx))
        else:
            self.fc_mix_mu = self.fc_mix_sigma = self.fc_mix_pi = None

        # ZI gate head
        if zi_idx:
            self.fc_gate = nn.Linear(hidden_dim, len(zi_idx))
        else:
            self.fc_gate = None

    def forward(self, z: torch.Tensor):
        B   = z.size(0)
        h0  = self.fc_h0(z).view(self.n_layers, B, self.hidden_dim)
        inp = self.fc_input(z).unsqueeze(1).expand(-1, self.window_size, -1)
        gru_out, _ = self.gru(inp, h0)           # (B, T, hidden)
        shared     = self.fc_shared(gru_out)      # (B, T, F) — raw logits

        # Build `out` without any inplace writes:
        #   Bernoulli features  → raw logit  (loss uses BCE-with-logits)
        #   everything else     → sigmoid    (keeps values in [0, 1])
        # We do this by selecting per-feature column without touching the result tensor.
        sig = torch.sigmoid(shared)   # (B, T, F)
        if self.bernoulli_idx:
            # Use torch.where with a broadcast mask to pick raw vs sigmoid, no inplace.
            bern_mask = torch.zeros(self.n_features, dtype=torch.bool,
                                    device=z.device)
            bern_mask[self.bernoulli_idx] = True
            bern_mask = bern_mask.view(1, 1, -1).expand_as(shared)
            out = torch.where(bern_mask, shared, sig)
        else:
            out = sig

        # Mixture heads (separate outputs, not in `out`)
        mix_mu = mix_sigma = mix_pi = None
        if self.fc_mix_mu is not None:
            mix_mu    = self.fc_mix_mu(gru_out)    # (B, T, |mix|)
            mix_sigma = self.fc_mix_sigma(gru_out)
            mix_pi    = self.fc_mix_pi(gru_out)

        gate_logit = self.fc_gate(gru_out) if self.fc_gate is not None else None
        return out, mix_mu, mix_sigma, mix_pi, gate_logit

    @torch.no_grad()
    def sample(self, z: torch.Tensor, bg_stats: dict) -> torch.Tensor:
        out, mix_mu, mix_sigma, mix_pi, gate_logit = self.forward(z)
        # Build list of per-feature columns, then stack — avoids all inplace ops.
        cols = [out[:, :, f].clone() for f in range(self.n_features)]

        # Bernoulli: sigmoid(raw_logit) → Bernoulli sample
        for f in self.bernoulli_idx:
            cols[f] = torch.bernoulli(torch.sigmoid(out[:, :, f]))

        # Mixture: sample from 2-component mixture
        if mix_mu is not None:
            for loc, g_idx in enumerate(self.mixture_idx):
                pi_tail     = torch.sigmoid(mix_pi[:, :, loc])
                use_tail    = torch.bernoulli(pi_tail).bool()
                sigma_t     = F.softplus(mix_sigma[:, :, loc]) + 1e-4
                tail_sample = mix_mu[:, :, loc] + sigma_t * torch.randn_like(sigma_t)
                mu_bg       = bg_stats.get(g_idx, {}).get('mu_bg', 0.5)
                sigma_bg    = bg_stats.get(g_idx, {}).get('sigma_bg', 0.2)
                bg_sample   = torch.randn_like(tail_sample) * sigma_bg + mu_bg
                cols[g_idx] = torch.where(use_tail, tail_sample, bg_sample).clamp(0, 1)

        # ZI gate
        if gate_logit is not None:
            gate_prob = torch.sigmoid(gate_logit)
            gate      = torch.bernoulli(gate_prob).bool()
            for s_local, s_global in enumerate(self.zi_idx):
                cols[s_global] = torch.where(
                    gate[:, :, s_local],
                    out[:, :, s_global],
                    torch.zeros_like(out[:, :, s_global])
                )

        return torch.stack(cols, dim=2)   # (B, T, F)


class BackgroundVAE(nn.Module):
    """Stage-1 VAE — learns shared background of all windows."""
    def __init__(self, n_features: int, window_size: int, hidden_dim: int,
                 latent_dim: int, bernoulli_idx: list, mixture_idx: list,
                 zi_idx: list, continuous_idx: list, n_layers: int = 1):
        super().__init__()
        self.latent_dim  = latent_dim
        self.mixture_idx = mixture_idx
        self.encoder     = REncoder(n_features, hidden_dim, latent_dim, n_layers)
        self.decoder     = MixtureDecoder(
            latent_dim, hidden_dim, n_features, window_size,
            bernoulli_idx, mixture_idx, zi_idx, continuous_idx, n_layers
        )

    def forward(self, x: torch.Tensor):
        mu, log_var           = self.encoder(x)
        z                     = self.encoder.reparameterize(mu, log_var)
        out, mix_mu, mix_sigma, mix_pi, gate_logit = self.decoder(z)
        return out, mix_mu, mix_sigma, mix_pi, gate_logit, mu, log_var

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu, _  = self.encoder(x)
        out, mix_mu, mix_sigma, mix_pi, _ = self.decoder(mu)
        # For mixture features, use the weighted mean as reconstruction
        if mix_mu is not None and self.decoder.fc_mix_mu is not None:
            for loc, g_idx in enumerate(self.mixture_idx):
                pi  = torch.sigmoid(mix_pi[:, :, loc])
                result_f = pi * mix_mu[:, :, loc] + (1 - pi) * out[:, :, g_idx]
                out[:, :, g_idx] = result_f
        return out

    @torch.no_grad()
    def sample(self, n: int, bg_stats: dict, device: str = 'cpu') -> np.ndarray:
        z   = torch.randn(n, self.latent_dim, device=device)
        out = self.decoder.sample(z, bg_stats)
        return out.cpu().numpy().astype(np.float32)


def _cyclical_kl_weight(epoch, n_epochs, n_cycles=4):
    cycle_len = max(n_epochs / n_cycles, 1)
    pos       = (epoch % cycle_len) / cycle_len
    return min(1.0, pos * 2.0)


def bg_vae_loss(x, out, mix_mu, mix_sigma, mix_pi, gate_logit, mu, log_var,
                bernoulli_idx, mixture_idx, zi_idx, continuous_idx,
                bg_stats, kl_weight, free_bits=1.0):
    """
    Mixed reconstruction loss:
      Bernoulli  → BCE-with-logits
      Mixture    → 2-component mixture NLL
      ZI-lognorm → gate BCE + lognormal NLL on non-zero values
      Continuous → MSE
    """
    recon   = torch.tensor(0.0, device=x.device)
    n_terms = 0

    # Bernoulli
    for f in bernoulli_idx:
        target = (x[:, :, f] > 0).float()
        recon += F.binary_cross_entropy_with_logits(out[:, :, f], target, reduction='mean')
        n_terms += 1

    # Mixture — 2-component mixture NLL
    if mix_mu is not None:
        for loc, g_idx in enumerate(mixture_idx):
            mu_bg    = bg_stats.get(g_idx, {}).get('mu_bg', 0.5)
            sigma_bg = bg_stats.get(g_idx, {}).get('sigma_bg', 0.2)
            recon   += mixture_nll(x[:, :, g_idx],
                                   mix_mu[:, :, loc],
                                   mix_sigma[:, :, loc],
                                   mu_bg, sigma_bg,
                                   mix_pi[:, :, loc])
            n_terms += 1

    # ZI-lognormal: gate BCE + lognormal NLL on non-zero part
    if gate_logit is not None:
        for s_local, s_global in enumerate(zi_idx):
            gate_label = (x[:, :, s_global] > 0).float()
            recon += 0.5 * F.binary_cross_entropy_with_logits(
                gate_logit[:, :, s_local], gate_label, reduction='mean')
            mask = gate_label.bool()
            if mask.any():
                # log-normal NLL: x is already log1p-normalised
                # treat as Gaussian in log-space → standard NLL
                val = x[:, :, s_global][mask]
                pred = out[:, :, s_global][mask].clamp(1e-6, 1 - 1e-6)
                # Use lognormal_nll with pred as mu, fixed log_sigma=0
                log_sigma = torch.zeros_like(pred)
                recon += 0.5 * lognormal_nll(val, pred, log_sigma)
            n_terms += 1

    # Continuous
    for f in continuous_idx:
        recon += F.mse_loss(out[:, :, f], x[:, :, f], reduction='mean')
        n_terms += 1

    recon_total    = float(x.shape[1] * x.shape[2]) * recon / max(n_terms, 1)
    kl_elem        = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_per_dim     = kl_elem.mean(dim=0)
    kl_loss        = torch.clamp(kl_per_dim, min=free_bits).sum()
    return recon_total + kl_weight * kl_loss, recon.item() / max(n_terms, 1), kl_loss.item()


def train_background_vae(X_np, window_size, n_features,
                         bernoulli_idx, mixture_idx, zi_idx, continuous_idx,
                         bg_stats, hidden_dim=128, latent_dim=32,
                         epochs=600, batch_size=256, lr=1e-3,
                         noise_std=0.05, free_bits=1.0, n_cycles=4,
                         device='cpu'):
    X   = torch.tensor(X_np.reshape(-1, window_size, n_features), device=device)
    mdl = BackgroundVAE(n_features, window_size, hidden_dim, latent_dim,
                        bernoulli_idx, mixture_idx, zi_idx, continuous_idx
                        ).to(device)
    opt    = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=1e-5)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X),
        batch_size=batch_size, shuffle=True)

    mdl.train()
    pbar = tqdm(range(epochs), desc='      BG-VAE ', unit='ep',
                dynamic_ncols=True, leave=False)
    for epoch in pbar:
        kl_w    = _cyclical_kl_weight(epoch, epochs, n_cycles)
        ep_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            opt.zero_grad()
            if noise_std > 0:
                noise = noise_std * torch.randn_like(batch)
                for f in bernoulli_idx:
                    noise[:, :, f] = 0.0
                batch_noisy = (batch + noise).clamp(0, 1)
            else:
                batch_noisy = batch
            out, mm, ms, mp, gate, mu_z, lv = mdl(batch_noisy)
            loss, _, _ = bg_vae_loss(
                batch, out, mm, ms, mp, gate, mu_z, lv,
                bernoulli_idx, mixture_idx, zi_idx, continuous_idx,
                bg_stats, kl_w, free_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 5.0)
            opt.step()
            ep_loss   += loss.item()
            n_batches += 1
        avg_loss = ep_loss / max(n_batches, 1)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', kl_w=f'{kl_w:.2f}')
    pbar.close()
    mdl.eval()
    return mdl


# ─────────────────────────────────────────────────────────────────────────────
# ATTACK-RESIDUAL VAE  (Stage 2)
# Trains on (attack_window − background_reconstruction) deltas.
# ─────────────────────────────────────────────────────────────────────────────

class ResidualVAE(nn.Module):
    """
    Simple GRU-VAE trained on attack residuals (delta = attack - bg_reconstruction).
    Uses lognormal NLL for mixture features (right-skewed residuals),
    MSE for rank features, BCE for binary features.
    """
    def __init__(self, n_features: int, window_size: int,
                 hidden_dim: int, latent_dim: int,
                 mixture_idx: list, continuous_idx: list,
                 bernoulli_idx: list, n_layers: int = 1):
        super().__init__()
        self.latent_dim   = latent_dim
        self.mixture_idx  = mixture_idx
        self.continuous_idx = continuous_idx
        self.bernoulli_idx  = bernoulli_idx
        self.encoder      = REncoder(n_features, hidden_dim, latent_dim, n_layers)
        self.fc_h0        = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_input     = nn.Linear(latent_dim, n_features)
        self.gru          = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True)
        self.fc_mu_out    = nn.Linear(hidden_dim, n_features)
        self.fc_sigma_out = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        mu_z, lv   = self.encoder(x)
        z          = self.encoder.reparameterize(mu_z, lv)
        B          = z.size(0)
        T          = x.size(1)
        h0         = self.fc_h0(z).view(self.encoder.gru.num_layers, B,
                                         self.encoder.gru.hidden_size)
        inp        = self.fc_input(z).unsqueeze(1).expand(-1, T, -1)
        gru_out, _ = self.gru(inp, h0)
        mu_out     = self.fc_mu_out(gru_out)
        sigma_out  = self.fc_sigma_out(gru_out)
        return mu_out, sigma_out, mu_z, lv

    @torch.no_grad()
    def sample(self, n: int, device='cpu') -> np.ndarray:
        z          = torch.randn(n, self.latent_dim, device=device)
        B          = n
        T          = 10  # will be fixed at init time
        h0         = self.fc_h0(z).view(self.encoder.gru.num_layers, B,
                                         self.encoder.gru.hidden_size)
        inp        = self.fc_input(z).unsqueeze(1).expand(-1, T, -1)
        gru_out, _ = self.gru(inp, h0)
        mu_out     = self.fc_mu_out(gru_out)
        sigma_out  = self.fc_sigma_out(gru_out)
        sigma      = F.softplus(sigma_out) + 1e-4
        samples    = mu_out + sigma * torch.randn_like(sigma)
        return samples.cpu().numpy().astype(np.float32)


class ResidualVAE2(nn.Module):
    """
    GRU-VAE for residuals — stores window_size at init so sample() works correctly.
    """
    def __init__(self, n_features: int, window_size: int,
                 hidden_dim: int, latent_dim: int, n_layers: int = 1):
        super().__init__()
        self.latent_dim  = latent_dim
        self.window_size = window_size
        self.n_features  = n_features
        self.n_layers    = n_layers
        self.encoder     = REncoder(n_features, hidden_dim, latent_dim, n_layers)
        self.fc_h0       = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_input    = nn.Linear(latent_dim, n_features)
        self.gru         = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True)
        self.fc_mu_out   = nn.Linear(hidden_dim, n_features)
        self.fc_sig_out  = nn.Linear(hidden_dim, n_features)
        self._hidden_dim = hidden_dim

    def forward(self, x):
        mu_z, lv   = self.encoder(x)
        z          = self.encoder.reparameterize(mu_z, lv)
        B          = z.size(0)
        h0         = self.fc_h0(z).view(self.n_layers, B, self._hidden_dim)
        inp        = self.fc_input(z).unsqueeze(1).expand(-1, self.window_size, -1)
        gru_out, _ = self.gru(inp, h0)
        return self.fc_mu_out(gru_out), self.fc_sig_out(gru_out), mu_z, lv

    @torch.no_grad()
    def sample(self, n: int, device='cpu') -> np.ndarray:
        z          = torch.randn(n, self.latent_dim, device=device)
        B          = n
        h0         = self.fc_h0(z).view(self.n_layers, B, self._hidden_dim)
        inp        = self.fc_input(z).unsqueeze(1).expand(-1, self.window_size, -1)
        gru_out, _ = self.gru(inp, h0)
        mu_out     = self.fc_mu_out(gru_out)
        sigma_out  = F.softplus(self.fc_sig_out(gru_out)) + 1e-4
        return (mu_out + sigma_out * torch.randn_like(sigma_out)).cpu().numpy().astype(np.float32)


def residual_loss(x, mu_out, sigma_out, mu_z, lv,
                  mixture_idx, kl_weight, free_bits=0.5):
    """
    Lognormal NLL for mixture features; MSE for rest.
    """
    sigma   = F.softplus(sigma_out) + 1e-4
    recon   = torch.tensor(0.0, device=x.device)
    n_terms = 0

    for f in mixture_idx:
        recon   += lognormal_nll(x[:, :, f], mu_out[:, :, f], sigma_out[:, :, f])
        n_terms += 1

    all_f = set(range(x.shape[2]))
    for f in all_f - set(mixture_idx):
        recon   += F.mse_loss(mu_out[:, :, f], x[:, :, f], reduction='mean')
        n_terms += 1

    recon_total = float(x.shape[1] * x.shape[2]) * recon / max(n_terms, 1)
    kl_elem     = -0.5 * (1 + lv - mu_z.pow(2) - lv.exp())
    kl_loss     = torch.clamp(kl_elem.mean(dim=0), min=free_bits).sum()
    return recon_total + kl_weight * kl_loss, recon.item() / max(n_terms, 1)


def train_residual_vae(delta_np, window_size, n_features, mixture_idx,
                       hidden_dim=64, latent_dim=16, epochs=600,
                       batch_size=128, lr=1e-3, free_bits=0.5,
                       n_cycles=4, device='cpu'):
    X   = torch.tensor(delta_np.reshape(-1, window_size, n_features), device=device)
    mdl = ResidualVAE2(n_features, window_size, hidden_dim, latent_dim).to(device)
    opt    = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=1e-5)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X), batch_size=batch_size, shuffle=True)
    mdl.train()
    pbar = tqdm(range(epochs), desc='      Residual', unit='ep',
                dynamic_ncols=True, leave=False)
    for epoch in pbar:
        kl_w      = _cyclical_kl_weight(epoch, epochs, n_cycles)
        ep_loss   = 0.0
        n_batches = 0
        for (batch,) in loader:
            opt.zero_grad()
            mu_out, sig_out, mu_z, lv = mdl(batch)
            loss, _ = residual_loss(batch, mu_out, sig_out, mu_z, lv,
                                    mixture_idx, kl_w, free_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 5.0)
            opt.step()
            ep_loss   += loss.item()
            n_batches += 1
        avg_loss = ep_loss / max(n_batches, 1)
        pbar.set_postfix(loss=f'{avg_loss:.4f}', kl_w=f'{kl_w:.2f}')
    pbar.close()
    mdl.eval()
    return mdl


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER  (LSTM — same as zirvae_multifile.py)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=64, n_layers=1, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def train_lstm(X_tr, y_tr, device='cpu', epochs=30, batch_size=128, lr=1e-3):
    F_  = X_tr.shape[2]
    clf = LSTMClassifier(F_).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    crt = nn.CrossEntropyLoss()
    Xt  = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt  = torch.tensor(y_tr, dtype=torch.long,    device=device)
    ldr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    clf.train()
    for _ in tqdm(range(epochs), desc='      LSTM clf', unit='ep',
                  dynamic_ncols=True, leave=False):
        for xb, yb in ldr:
            opt.zero_grad(); crt(clf(xb), yb).backward(); opt.step()
    clf.eval()
    return clf


def eval_lstm(clf, X, y, device='cpu'):
    from sklearn.metrics import f1_score
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = clf(Xt).argmax(dim=1).cpu().numpy()
    return float(f1_score(y, preds, average='macro', zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def ks_per_class(real_X, real_y, syn_X, syn_y, feat_cols):
    results = {}
    for cls in [0, 1]:
        r_flat = real_X[real_y == cls].mean(axis=1)
        s_flat = syn_X[syn_y == cls].mean(axis=1)
        rows   = []
        for i, col in enumerate(feat_cols):
            s, p = scipy_stats.ks_2samp(r_flat[:, i], s_flat[:, i])
            rows.append({'feature': col, 'ks_stat': round(float(s), 4),
                         'p_value': round(float(p), 4), 'similar': bool(p > 0.05)})
        results[cls] = pd.DataFrame(rows).sort_values('ks_stat', ascending=False)
    return results


def tail_ks(real_X, real_y, syn_X, syn_y, feat_cols, tail_thresh_idx: int,
            quantile: float = 0.90):
    """
    KS test restricted to windows where the tail feature mean > quantile of normal.
    """
    normal_vals = real_X[real_y == 0, :, tail_thresh_idx].mean(axis=1)
    thresh      = float(np.quantile(normal_vals, quantile))

    def _tail_mask(X, y):
        vals = X[:, :, tail_thresh_idx].mean(axis=1)
        return (y == 1) & (vals > thresh)

    real_tail = real_X[_tail_mask(real_X, real_y)]
    syn_tail  = syn_X[_tail_mask(syn_X, syn_y)]

    if len(real_tail) < 5 or len(syn_tail) < 5:
        return pd.DataFrame({'feature': feat_cols,
                             'tail_ks': [float('nan')] * len(feat_cols),
                             'tail_p':  [float('nan')] * len(feat_cols)})
    rows = []
    r_flat = real_tail.mean(axis=1)
    s_flat = syn_tail.mean(axis=1)
    for i, col in enumerate(feat_cols):
        s, p = scipy_stats.ks_2samp(r_flat[:, i], s_flat[:, i])
        rows.append({'feature': col, 'tail_ks': round(float(s), 4),
                     'tail_p': round(float(p), 4)})
    return pd.DataFrame(rows)


def tail_coverage(real_X, real_y, syn_X, syn_y,
                  feat_cols, tail_thresh_idx: int, quantile: float = 0.90):
    """
    For each feature: fraction of REAL extreme-attack values that fall within
    [min, max] of SYNTHETIC extreme-attack values.
    Coverage = 1 → synthetic fully spans the real tail.
    Coverage < 0.5 → synthetic severely under-estimates the tail.
    """
    normal_vals = real_X[real_y == 0, :, tail_thresh_idx].mean(axis=1)
    thresh      = float(np.quantile(normal_vals, quantile))

    def _tail_flat(X, y):
        mask = (y == 1) & (X[:, :, tail_thresh_idx].mean(axis=1) > thresh)
        return X[mask].mean(axis=1)   # (N_tail, F)

    r = _tail_flat(real_X, real_y)
    s = _tail_flat(syn_X,  syn_y)

    rows = []
    for i, col in enumerate(feat_cols):
        if len(r) == 0 or len(s) == 0:
            rows.append({'feature': col, 'tail_coverage': float('nan')})
            continue
        r_col  = r[:, i]
        s_min  = s[:, i].min()
        s_max  = s[:, i].max()
        cov    = float(((r_col >= s_min) & (r_col <= s_max)).mean())
        rows.append({'feature': col, 'tail_coverage': round(cov, 4)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(real_X, real_y, syn_X, syn_y,
                       feat_cols, variant, out_dir):
    F_ = len(feat_cols)
    fig, axes = plt.subplots(2, F_, figsize=(2.8 * F_, 7), constrained_layout=True)
    for row, cls in enumerate([0, 1]):
        label   = 'Normal' if cls == 0 else 'Attack'
        r_flat  = real_X[real_y == cls].mean(axis=1)
        s_flat  = syn_X[syn_y == cls].mean(axis=1)
        for col_idx, col in enumerate(feat_cols):
            ax = axes[row, col_idx]
            r  = r_flat[:, col_idx]
            s  = s_flat[:, col_idx]
            ax.hist(r, bins=30, alpha=0.4, color='steelblue', density=True, label='Real')
            ax.hist(s, bins=30, alpha=0.4, color='tomato',    density=True, label='Syn')
            try:
                xs = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 120)
                ax.plot(xs, scipy_stats.gaussian_kde(r)(xs), 'steelblue', lw=1.2)
                ax.plot(xs, scipy_stats.gaussian_kde(s)(xs), 'tomato',    lw=1.2)
            except Exception:
                pass
            ks_v, _ = scipy_stats.ks_2samp(r, s)
            ax.set_title(f'{col}\nKS={ks_v:.3f}', fontsize=6)
            ax.tick_params(labelsize=5)
            if col_idx == 0:
                ax.set_ylabel(f'Class {cls} ({label})', fontsize=7)
            if row == 0 and col_idx == 0:
                ax.legend(fontsize=5)
    fig.suptitle(f'Distribution Overlay — {variant}', fontsize=10)
    path = os.path.join(out_dir, f'distributions_{variant}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    return path


def plot_ks_bars(ks_results, variant, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for ax, cls in zip(axes, [0, 1]):
        df  = ks_results[cls].sort_values('ks_stat', ascending=False)
        clr = ['steelblue' if r else 'tomato' for r in df['similar']]
        ax.bar(df['feature'], df['ks_stat'], color=clr)
        ax.axhline(0.05, color='black', linestyle='--', lw=1.2)
        label = 'Normal' if cls == 0 else 'Attack'
        ax.set_title(f'Class {cls} ({label})  '
                     f'mean KS={df["ks_stat"].mean():.3f}  '
                     f'pass={df["similar"].sum()}/{len(df)}')
        ax.set_ylabel('KS Statistic'); ax.set_ylim(0, 1.05)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    fig.suptitle(f'KS per feature — {variant}  (blue=pass, red=fail)', fontsize=10)
    path = os.path.join(out_dir, f'ks_{variant}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    return path


def plot_tail_coverage(tail_cov_df, variant, out_dir):
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    colors  = ['seagreen' if v >= 0.8 else ('orange' if v >= 0.5 else 'tomato')
               for v in tail_cov_df['tail_coverage'].fillna(0)]
    ax.bar(tail_cov_df['feature'], tail_cov_df['tail_coverage'], color=colors)
    ax.axhline(0.8, color='green',  linestyle='--', lw=1, label='0.8 target')
    ax.axhline(0.5, color='orange', linestyle='--', lw=1, label='0.5 threshold')
    ax.set_ylim(0, 1.05); ax.set_ylabel('Tail Coverage')
    ax.set_title(f'Tail Coverage (90th pct) — {variant}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=7)
    path = os.path.join(out_dir, f'tail_coverage_{variant}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    return path


def plot_tstr_summary(summary_rows, out_dir):
    df   = pd.DataFrame(summary_rows)
    if df.empty:
        return
    variants = df['variant'].tolist()
    x        = np.arange(len(variants))
    width    = 0.25
    fig, ax  = plt.subplots(figsize=(max(10, len(variants)), 5), constrained_layout=True)
    ax.bar(x - width, df['baseline_f1'], width, label='Baseline (real→real)', color='steelblue',  alpha=0.85)
    ax.bar(x,         df['tstr_f1'],     width, label='TSTR (syn→real)',       color='tomato',     alpha=0.85)
    ax.bar(x + width, df['trts_f1'],     width, label='TRTS (real→syn)',       color='seagreen',   alpha=0.85)
    for i, row in df.iterrows():
        ax.text(i - width, row['baseline_f1'] + 0.01, f"{row['baseline_f1']:.2f}", ha='center', fontsize=6)
        ax.text(i,         row['tstr_f1']     + 0.01, f"{row['tstr_f1']:.2f}",     ha='center', fontsize=6)
        ax.text(i + width, row['trts_f1']     + 0.01, f"{row['trts_f1']:.2f}",     ha='center', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Macro F1')
    ax.axhline(0.5, color='red', linestyle='--', lw=1, label='random')
    ax.legend(fontsize=8)
    ax.set_title('TSTR / TRTS / Baseline — All Blackhole Variants')
    path = os.path.join(out_dir, 'tstr_trts_all_variants.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    return path


def plot_mean_tail_ks(summary_rows, out_dir):
    df = pd.DataFrame(summary_rows)
    if df.empty or 'mean_tail_ks' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(df)), 4), constrained_layout=True)
    colors = ['seagreen' if v < 0.1 else ('orange' if v < 0.3 else 'tomato')
              for v in df['mean_tail_ks'].fillna(1.0)]
    ax.bar(df['variant'], df['mean_tail_ks'], color=colors)
    ax.axhline(0.1, color='green',  linestyle='--', lw=1, label='good (<0.1)')
    ax.axhline(0.3, color='orange', linestyle='--', lw=1, label='marginal (<0.3)')
    ax.set_ylim(0, 1.05); ax.set_ylabel('Mean Tail KS')
    ax.set_title('Mean Tail KS (90th pct, attack class) — All Blackhole Variants')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=7)
    path = os.path.join(out_dir, 'mean_tail_ks_all_variants.png')
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PER-VARIANT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(variant: str) -> dict:
    tqdm.write(f"\n{'='*65}")
    tqdm.write(f"  BLACKHOLE VARIANT: {variant}")
    tqdm.write(f"{'='*65}")

    folder_path = os.path.join(DATA_ROOT, variant)
    out_dir     = os.path.join(OUT_ROOT, variant)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load & preprocess ────────────────────────────────────────────────────
    df_tr, df_te = split_files(folder_path)
    X_tr, y_tr, X_te, y_te, feat_cols, preproc = preprocess_and_window(
        df_tr, df_te, WINDOW_SIZE)
    T, F_ = X_tr.shape[1], X_tr.shape[2]

    tqdm.write(f"  Windows — Train: {X_tr.shape}  Test: {X_te.shape}")
    tqdm.write(f"  Label dist train: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    # ── Feature index classification ─────────────────────────────────────────
    bernoulli_idx = [i for i, c in enumerate(feat_cols)
                     if c in ('disr', 'diss', 'disr.1', 'diss.1')]
    mixture_idx   = [i for i, c in enumerate(feat_cols)
                     if c in MIXTURE_FEATS and c not in ('disr', 'diss', 'disr.1', 'diss.1')]
    zi_idx        = [i for i, c in enumerate(feat_cols)
                     if c in ('diar', 'diar.1')]
    all_typed     = set(bernoulli_idx + mixture_idx + zi_idx)
    continuous_idx = [i for i in range(F_) if i not in all_typed]

    tqdm.write(f"  bernoulli : {[feat_cols[i] for i in bernoulli_idx]}")
    tqdm.write(f"  mixture   : {[feat_cols[i] for i in mixture_idx]}")
    tqdm.write(f"  zi_lognorm: {[feat_cols[i] for i in zi_idx]}")
    tqdm.write(f"  continuous: {[feat_cols[i] for i in continuous_idx]}")

    # Background stats for mixture decoder (fitted on training normal class)
    bg_stats = {}
    X_tr_normal = X_tr[y_tr == 0]
    for g_idx in mixture_idx:
        vals = X_tr_normal[:, :, g_idx].flatten()
        bg_stats[g_idx] = {
            'mu_bg':    float(vals.mean()),
            'sigma_bg': float(max(vals.std(), 1e-4)),
        }

    # ── Hyperparams ──────────────────────────────────────────────────────────
    latent_dim = max(8, min((T * F_) // 8, 32))
    hidden_dim = max(128, F_ * 8)
    batch_size = min(512, max(64, len(X_tr) // 10))
    free_bits  = round(min(0.5, 8.0 / latent_dim), 4)

    tqdm.write(f"  latent={latent_dim}  hidden={hidden_dim}  "
               f"batch={batch_size}  free_bits={free_bits}")

    # ── Extreme-quantile flag ─────────────────────────────────────────────────
    tots_idx    = feat_cols.index('tots') if 'tots' in feat_cols else mixture_idx[0]
    X_tr_aug, feat_cols_aug, tail_thresh = add_extreme_flag(
        X_tr, y_tr, feat_cols, TAIL_QUANTILE)
    X_te_aug, _, _ = add_extreme_flag(X_te, y_te, feat_cols, TAIL_QUANTILE)
    F_aug = X_tr_aug.shape[2]
    continuous_idx_aug = continuous_idx + [F_aug - 1]
    mixture_idx_aug    = mixture_idx

    tqdm.write(f"  Tail threshold (tots 90th pct of normal): {tail_thresh:.4f}")

    # ── BASELINE LSTM ─────────────────────────────────────────────────────────
    tqdm.write(f"\n  [1/4] {variant}: Training baseline LSTM …")
    base_clf = train_lstm(X_tr, y_tr, device=DEVICE)
    base_f1  = eval_lstm(base_clf, X_te, y_te, device=DEVICE)
    tqdm.write(f"       Baseline F1 = {base_f1:.4f}")

    # ── STAGE 1 — BACKGROUND VAE (all windows, augmented features) ────────────
    tqdm.write(f"\n  [2/4] {variant}: Stage-1 Background VAE ({BG_EPOCHS} epochs) …")
    bernoulli_idx_aug = bernoulli_idx   # same indices, aug doesn't change them
    zi_idx_aug        = zi_idx
    bg_vae = train_background_vae(
        X_np          = X_tr_aug,
        window_size   = T,
        n_features    = F_aug,
        bernoulli_idx = bernoulli_idx_aug,
        mixture_idx   = mixture_idx_aug,
        zi_idx        = zi_idx_aug,
        continuous_idx = continuous_idx_aug,
        bg_stats      = bg_stats,
        hidden_dim    = hidden_dim,
        latent_dim    = latent_dim,
        epochs        = BG_EPOCHS,
        batch_size    = batch_size,
        free_bits     = free_bits,
        device        = DEVICE,
    )

    # Reconstruct attack windows → compute deltas
    X_atk_aug   = torch.tensor(X_tr_aug[y_tr == 1], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        X_bg_recon  = bg_vae.reconstruct(X_atk_aug).cpu().numpy()  # (N_atk, T, F_aug)
    X_atk_np    = X_tr_aug[y_tr == 1]
    delta       = X_atk_np - X_bg_recon                            # attack residual

    # ── STAGE 2 — RESIDUAL VAE (attack deltas only) ───────────────────────────
    tqdm.write(f"\n  [3/4] {variant}: Stage-2 Residual VAE ({DELTA_EPOCHS} epochs, n_delta={len(delta)}) …")
    res_vae = train_residual_vae(
        delta_np    = delta,
        window_size = T,
        n_features  = F_aug,
        mixture_idx = mixture_idx_aug,
        hidden_dim  = max(64, hidden_dim // 2),
        latent_dim  = max(8, latent_dim // 2),
        epochs      = DELTA_EPOCHS,
        batch_size  = max(32, min(batch_size, len(delta) // 4)),
        free_bits   = free_bits,
        device      = DEVICE,
    )

    # ── GENERATE SYNTHETIC DATA ───────────────────────────────────────────────
    tqdm.write(f"\n  [4/4] {variant}: Generating & evaluating synthetic windows …")
    # Normal class: sample from Background VAE, set is_extreme=0
    bg_syn_aug = bg_vae.sample(N_SYNTH_PER_CLASS, bg_stats, device=DEVICE)
    bg_syn_aug[:, :, -1] = 0.0   # is_extreme = 0 for normal

    # Attack class: sample background + sample residual, clamp
    bg_atk_aug  = bg_vae.sample(N_SYNTH_PER_CLASS, bg_stats, device=DEVICE)
    res_delta   = res_vae.sample(N_SYNTH_PER_CLASS, device=DEVICE)
    syn_atk_aug = np.clip(bg_atk_aug + res_delta, 0.0, 1.0)

    # Strip the is_extreme flag (last column) before evaluation
    syn_normal = bg_syn_aug[:, :, :F_]
    syn_attack = syn_atk_aug[:, :, :F_]

    X_syn_all  = np.concatenate([syn_normal, syn_attack], axis=0)
    y_syn_all  = np.concatenate([
        np.zeros(N_SYNTH_PER_CLASS, dtype=np.int64),
        np.ones( N_SYNTH_PER_CLASS, dtype=np.int64)
    ])

    # ── EVALUATE ──────────────────────────────────────────────────────────────
    ks_res  = ks_per_class(X_tr, y_tr, X_syn_all, y_syn_all, feat_cols)
    tstr_f1 = eval_lstm(train_lstm(X_syn_all, y_syn_all, device=DEVICE),
                        X_te, y_te, device=DEVICE)
    trts_f1 = eval_lstm(train_lstm(X_tr, y_tr, device=DEVICE),
                        X_syn_all, y_syn_all, device=DEVICE)

    tail_ks_df  = tail_ks(X_tr, y_tr, X_syn_all, y_syn_all,
                           feat_cols, tots_idx, TAIL_QUANTILE)
    tail_cov_df = tail_coverage(X_tr, y_tr, X_syn_all, y_syn_all,
                                feat_cols, tots_idx, TAIL_QUANTILE)

    # Conditional precision: real classifier scores synthetic attacks
    syn_atk_probs = []
    clf_real = train_lstm(X_tr, y_tr, device=DEVICE)
    Xt_syn_atk = torch.tensor(syn_attack, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = clf_real(Xt_syn_atk)
        preds  = logits.argmax(dim=1).cpu().numpy()
    cond_precision = float((preds == 1).mean())

    # Print summary
    for cls in [0, 1]:
        kdf   = ks_res[cls]
        label = 'Normal' if cls == 0 else 'Attack'
        tqdm.write(f"  KS cls{cls} ({label}): pass={kdf['similar'].sum()}/{len(kdf)}"
                   f"  mean_KS={kdf['ks_stat'].mean():.4f}")
    tqdm.write(f"  TSTR={tstr_f1:.4f}  TRTS={trts_f1:.4f}  Baseline={base_f1:.4f}")
    tqdm.write(f"  Mean Tail KS:       {tail_ks_df['tail_ks'].mean():.4f}")
    tqdm.write(f"  Mean Tail Coverage: {tail_cov_df['tail_coverage'].mean():.4f}")
    tqdm.write(f"  Cond. Precision:    {cond_precision:.4f}")

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    plot_distributions(X_tr, y_tr, X_syn_all, y_syn_all, feat_cols, variant, out_dir)
    plot_ks_bars(ks_res, variant, out_dir)
    plot_tail_coverage(tail_cov_df, variant, out_dir)

    # ── SAVE METRICS ──────────────────────────────────────────────────────────
    metrics = {
        'variant':          variant,
        'baseline_f1':      round(base_f1, 4),
        'tstr_f1':          round(tstr_f1, 4),
        'trts_f1':          round(trts_f1, 4),
        'cond_precision':   round(cond_precision, 4),
        'mean_tail_ks':     round(float(tail_ks_df['tail_ks'].mean()), 4),
        'mean_tail_coverage': round(float(tail_cov_df['tail_coverage'].mean()), 4),
        'ks_cls0_mean':     round(float(ks_res[0]['ks_stat'].mean()), 4),
        'ks_cls1_mean':     round(float(ks_res[1]['ks_stat'].mean()), 4),
        'ks_cls0_pass':     int(ks_res[0]['similar'].sum()),
        'ks_cls1_pass':     int(ks_res[1]['similar'].sum()),
        'tail_thresh':      round(tail_thresh, 4),
        'ks_per_feature_cls0': ks_res[0].to_dict(orient='records'),
        'ks_per_feature_cls1': ks_res[1].to_dict(orient='records'),
        'tail_ks':          tail_ks_df.to_dict(orient='records'),
        'tail_coverage':    tail_cov_df.to_dict(orient='records'),
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Running two-stage blackhole VAE on {len(VARIANTS)} variants …")
    all_metrics = []

    outer = tqdm(VARIANTS, desc='Overall progress', unit='variant',
                 dynamic_ncols=True, position=0)
    for variant in outer:
        outer.set_description(f'Overall [{variant}]')
        try:
            m = run_variant(variant)
            all_metrics.append(m)
            outer.set_postfix(tstr=f"{m['tstr_f1']:.3f}",
                              tail_ks=f"{m['mean_tail_ks']:.3f}")
        except Exception as e:
            import traceback
            print(f"  ERROR in {variant}: {e}")
            traceback.print_exc()

    if not all_metrics:
        print("No variants processed.")
        return

    # Save global summary
    summary_path = os.path.join(OUT_ROOT, 'summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(all_metrics, fh, indent=2)

    summary_df = pd.DataFrame([{
        k: v for k, v in m.items()
        if not isinstance(v, list)
    } for m in all_metrics])
    summary_df.to_csv(os.path.join(OUT_ROOT, 'summary.csv'), index=False)

    # Cross-variant TSTR/TRTS plot
    plot_tstr_summary(all_metrics, OUT_ROOT)
    plot_mean_tail_ks(all_metrics, OUT_ROOT)

    # Console summary table
    print('\n' + '='*75)
    print('  FINAL SUMMARY — Two-Stage Blackhole VAE')
    print('='*75)
    hdr = f"  {'Variant':<28} {'Base':>6} {'TSTR':>6} {'TRTS':>6} "
    hdr += f"{'TailKS':>7} {'TailCov':>8} {'CondPrec':>9}"
    print(hdr); print('-' * len(hdr))
    for m in all_metrics:
        print(f"  {m['variant']:<28} {m['baseline_f1']:>6.3f} "
              f"{m['tstr_f1']:>6.3f} {m['trts_f1']:>6.3f} "
              f"{m['mean_tail_ks']:>7.3f} {m['mean_tail_coverage']:>8.3f} "
              f"{m['cond_precision']:>9.3f}")

    print(f"\nAll results saved → {OUT_ROOT}")
    print(f"Summary JSON      → {summary_path}")


if __name__ == '__main__':
    main()
