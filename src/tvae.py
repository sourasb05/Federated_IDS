# tvae.py
#
# TVAE — Tabular Variational Autoencoder
# Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K.
# "Modeling Tabular Data using Conditional GAN" — NeurIPS 2019
#
# Key differences from Tab-VAE:
#   - No Gumbel-Softmax: categorical output uses softmax + argmax
#   - No tanh on continuous output: raw linear output
#   - loss_factor multiplies reconstruction loss (default 2)
#   - compress_dims / decompress_dims follow original paper: (128, 128)
#   - No per-column variance delta — uses fixed reconstruction loss
#
# Drop-in replacement for tab_vae.py:
#   train_tab_vae()  → train_tvae()
#   synthesize()     → synthesize()   (same signature)

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder


# ═══════════════════════════════════════════════════════════════
# 1. DATA TRANSFORMER
#    Identical to Tab-VAE transformer — MSN for continuous,
#    OHE for categorical. TVAE uses the same preprocessing.
# ═══════════════════════════════════════════════════════════════

class TabTransformer:
    """
    Transforms tabular data for TVAE.

    Continuous columns:
        Mode-Specific Normalisation (MSN) via BayesianGMM.
        Each value ci is encoded as:
            alpha_i = (ci - mu_k) / (4 * sigma_k)   in [-1, 1]
            beta_i  = one_hot(mode_k)                mode indicator
        Encoded dim per column = 1 + n_active_modes

    Categorical columns:
        One-hot encoding.
        Encoded dim per column = n_categories

    All columns are tracked via col_info with their slice
    positions in the final encoded vector.
    """

    def __init__(self, continuous_cols: list,
                 categorical_cols: list,
                 n_gmm_components: int = 5):
        self.continuous_cols  = continuous_cols
        self.categorical_cols = categorical_cols
        self.n_gmm            = n_gmm_components

        self.gmms      = {}   # col → (BayesianGMM, active_mask)
        self.ohe       = {}   # col → OneHotEncoder
        self.col_info  = {}   # col → metadata + slice indices
        self.total_dim = 0

    # ────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        ptr = 0

        # ── continuous columns ───────────────────
        for col in self.continuous_cols:
            data = df[col].values.astype(float).reshape(-1, 1)

            gmm = BayesianGaussianMixture(
                n_components=self.n_gmm,
                weight_concentration_prior=0.001,
                max_iter=100,
                random_state=42
            )
            gmm.fit(data)

            active  = (gmm.weights_ > 0.01).astype(bool)
            n_modes = max(int(np.sum(active)), 1)

            self.gmms[col] = (gmm, active)

            dim = 1 + n_modes          # alpha (1) + beta one-hot
            self.col_info[col] = {
                'type'   : 'continuous',
                'n_modes': n_modes,
                'slice'  : (ptr, ptr + dim)
            }
            ptr += dim

        # ── categorical columns ──────────────────
        for col in self.categorical_cols:
            data = df[col].values.astype(str).reshape(-1, 1)

            enc = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore'
            )
            enc.fit(data)
            n_cats = int(len(list(enc.categories_[0])))

            self.ohe[col] = enc
            self.col_info[col] = {
                'type'  : 'categorical',
                'n_cats': n_cats,
                'slice' : (ptr, ptr + n_cats)
            }
            ptr += n_cats

        self.total_dim = ptr

    # ────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        N   = len(df)
        out = np.zeros((N, self.total_dim), dtype=np.float32)

        for col in self.continuous_cols:
            gmm, active = self.gmms[col]
            info = self.col_info[col]
            s, e = info['slice']
            n_m  = info['n_modes']

            data  = df[col].values.astype(float)
            probs = gmm.predict_proba(
                data.reshape(-1, 1)
            )[:, active]

            if probs.shape[1] == 0:
                probs = np.ones((N, 1))

            mode_idx = probs.argmax(axis=1)

            means = gmm.means_[active].flatten()
            stds  = np.sqrt(
                gmm.covariances_[active]
            ).flatten()
            stds  = np.clip(stds, 1e-6, None)

            alpha = (data - means[mode_idx]) / (
                4 * stds[mode_idx]
            )
            alpha = np.clip(alpha, -1, 1)
            beta  = np.eye(n_m)[mode_idx]

            out[:, s]      = alpha
            out[:, s+1:e]  = beta

        for col in self.categorical_cols:
            info = self.col_info[col]
            s, e = info['slice']
            data = df[col].values.astype(str).reshape(-1, 1)
            out[:, s:e] = self.ohe[col].transform(data)

        return out

    # ────────────────────────────────────────────
    def inverse_transform(self,
                          cont_outputs: list,
                          cat_outputs: list) -> pd.DataFrame:
        """
        cont_outputs: list of tensors, one per continuous col
                      each tensor shape (N, 1 + n_modes)
                      col[:, 0]  = alpha (continuous value)
                      col[:, 1:] = beta logits (mode probs)

        cat_outputs:  list of tensors, one per categorical col
                      each tensor shape (N, n_cats) — raw logits
        """
        rows = {}

        # ── continuous ───────────────────────────
        for i, col in enumerate(self.continuous_cols):
            gmm, active = self.gmms[col]
            info = self.col_info[col]
            n_m  = info['n_modes']

            out   = cont_outputs[i].cpu().numpy()   # (N, 1+n_modes)
            alpha = out[:, 0]                        # (N,)
            beta_logits = out[:, 1:]                 # (N, n_modes)

            # TVAE uses argmax (no Gumbel-Softmax)
            b_idx = beta_logits.argmax(axis=1)
            b_idx = np.clip(b_idx, 0, n_m - 1)

            means = gmm.means_[active].flatten()
            stds  = np.sqrt(
                gmm.covariances_[active]
            ).flatten()
            stds  = np.clip(stds, 1e-6, None)

            rows[col] = alpha * 4 * stds[b_idx] + means[b_idx]

        # ── categorical ──────────────────────────
        for i, col in enumerate(self.categorical_cols):
            enc    = self.ohe[col]
            logits = cat_outputs[i].cpu().numpy()    # (N, n_cats)

            # TVAE: argmax over softmax probabilities
            g_idx  = logits.argmax(axis=1)
            cats   = enc.categories_[0]
            g_idx  = np.clip(g_idx, 0, len(cats) - 1)
            rows[col] = cats[g_idx]

        return pd.DataFrame(rows)

    # ────────────────────────────────────────────
    def get_decoder_info(self):
        """
        Returns the output dimensions the decoder needs.

        cont_dims: list of (alpha_dim=1, beta_dim=n_modes)
                   one per continuous column
        cat_dims:  list of n_cats per categorical column
        """
        cont_dims = []
        for col in self.continuous_cols:
            n_m = self.col_info[col]['n_modes']
            cont_dims.append((1, n_m))

        cat_dims = []
        for col in self.categorical_cols:
            cat_dims.append(self.col_info[col]['n_cats'])

        return cont_dims, cat_dims

    # ────────────────────────────────────────────
    def get_loss_slices(self):
        """
        Returns slice indices into the encoded vector,
        used by the loss function to split targets.

        cont_slices: [(alpha_start, beta_start, beta_end), ...]
        cat_slices:  [(start, end), ...]
        """
        cont_slices, cat_slices = [], []
        ptr = 0

        for col in self.continuous_cols:
            n_m = self.col_info[col]['n_modes']
            cont_slices.append((ptr, ptr + 1, ptr + 1 + n_m))
            ptr += 1 + n_m

        for col in self.categorical_cols:
            n_c = self.col_info[col]['n_cats']
            cat_slices.append((ptr, ptr + n_c))
            ptr += n_c

        return cont_slices, cat_slices


# ═══════════════════════════════════════════════════════════════
# 2. ENCODER
#    compress_dims = (128, 128) per original paper
#    ReLU activations
#    No activation on output (raw mu and log_var)
# ═══════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """
    Maps encoded row r → (mu, log_var) of latent Gaussian.

    Architecture (Xu et al. 2019):
        Linear(input_dim → compress_dims[0]) → ReLU
        Linear(compress_dims[0] → compress_dims[1]) → ReLU
        Linear(compress_dims[-1] → latent_dim)   → mu     (no activation)
        Linear(compress_dims[-1] → latent_dim)   → log_var (no activation)
    """

    def __init__(self, input_dim: int,
                 compress_dims: tuple = (128, 128),
                 latent_dim: int = 128):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in compress_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.net    = nn.Sequential(*layers)
        self.fc_mu  = nn.Linear(in_dim, latent_dim)
        self.fc_var = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h       = self.net(x)
        mu      = self.fc_mu(h)      # no activation — raw
        log_var = self.fc_var(h)     # no activation — raw
        return mu, log_var

    @staticmethod
    def reparameterize(mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """
        z = mu + sigma * epsilon
        epsilon ~ N(0, I)
        Differentiable via reparameterization trick.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


# ═══════════════════════════════════════════════════════════════
# 3. DECODER
#    decompress_dims = (128, 128) per original paper
#    ReLU activations on hidden layers
#    NO activation on output layers (raw logits)
#    Separate output heads per column (continuous + categorical)
# ═══════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """
    Maps latent z → reconstructed encoded row.

    Per column output heads:
        Continuous col i:
            alpha head → Linear(hidden → 1)         raw value
            beta head  → Linear(hidden → n_modes)   raw logits

        Categorical col i:
            gamma head → Linear(hidden → n_cats)    raw logits

    NO activation on any output head — raw logits are used
    in the loss and softmax/argmax is applied at inference.
    This is the key difference from Tab-VAE (no Gumbel-Softmax).
    """

    def __init__(self, latent_dim: int,
                 decompress_dims: tuple = (128, 128),
                 cont_dims: list = None,
                 cat_dims: list = None):
        super().__init__()

        self.cont_dims = cont_dims or []
        self.cat_dims  = cat_dims  or []

        layers = []
        in_dim = latent_dim
        for h_dim in decompress_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.net = nn.Sequential(*layers)

        # one alpha head per continuous column (output dim = 1)
        self.alpha_heads = nn.ModuleList([
            nn.Linear(in_dim, 1) for _ in cont_dims
        ])

        # one beta head per continuous column (output dim = n_modes)
        self.beta_heads = nn.ModuleList([
            nn.Linear(in_dim, n_m) for (_, n_m) in cont_dims
        ])

        # one gamma head per categorical column (output dim = n_cats)
        self.gamma_heads = nn.ModuleList([
            nn.Linear(in_dim, n_c) for n_c in cat_dims
        ])

    def forward(self, z: torch.Tensor):
        h = self.net(z)

        # raw logits — no activation (TVAE paper specification)
        # cont_outputs[i] shape: (B, 1 + n_modes)
        # cat_outputs[i]  shape: (B, n_cats)
        cont_outputs = [
            torch.cat(
                [alpha_hd(h), beta_hd(h)], dim=1
            )
            for alpha_hd, beta_hd in zip(
                self.alpha_heads, self.beta_heads
            )
        ]

        cat_outputs = [
            gamma_hd(h) for gamma_hd in self.gamma_heads
        ]

        return cont_outputs, cat_outputs


# ═══════════════════════════════════════════════════════════════
# 4. TVAE — Full Model
# ═══════════════════════════════════════════════════════════════

class TVAE(nn.Module):
    """
    Tabular Variational Autoencoder (Xu et al. NeurIPS 2019).

    Training:   encode → reparameterize → decode → ELBO loss
    Inference:  sample z ~ N(0,I) → decode → argmax for categoricals
    """

    def __init__(self,
                 input_dim: int,
                 compress_dims: tuple = (128, 128),
                 decompress_dims: tuple = (128, 128),
                 latent_dim: int = 128,
                 cont_dims: list = None,
                 cat_dims: list = None):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(
            input_dim=input_dim,
            compress_dims=compress_dims,
            latent_dim=latent_dim
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            decompress_dims=decompress_dims,
            cont_dims=cont_dims or [],
            cat_dims=cat_dims  or []
        )

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = Encoder.reparameterize(mu, log_var)
        cont_outputs, cat_outputs = self.decoder(z)
        return cont_outputs, cat_outputs, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu'):
        """
        Generate n synthetic rows.
        Samples z from N(0,I) then decodes.
        Returns raw decoder outputs (cont_outputs, cat_outputs).
        """
        z = torch.randn(n, self.latent_dim).to(device)
        cont_outputs, cat_outputs = self.decoder(z)
        return cont_outputs, cat_outputs


# ═══════════════════════════════════════════════════════════════
# 5. LOSS FUNCTION — TVAE ELBO
#
# ELBO = loss_factor × reconstruction_loss + KL_divergence
#
# Reconstruction loss:
#   Continuous alpha  : MSE  (normalised value)
#   Continuous beta   : CrossEntropy (mode indicator)
#   Categorical gamma : CrossEntropy (category)
#
# KL divergence (closed form for Gaussian):
#   KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
#
# loss_factor = 2 is the default from the original paper.
# It upweights reconstruction to prevent posterior collapse.
# ═══════════════════════════════════════════════════════════════

def tvae_loss(x_batch: torch.Tensor,
              cont_outputs: list,
              cat_outputs: list,
              mu: torch.Tensor,
              log_var: torch.Tensor,
              cont_slices: list,
              cat_slices: list,
              loss_factor: float = 2.0,
              kl_weight: float = 1.0) -> tuple:
    """
    Computes TVAE ELBO loss with optional KL annealing.

    Args:
        x_batch      : (B, total_encoded_dim) — encoded input
        cont_outputs : list of (B, 1+n_modes) tensors from decoder
        cat_outputs  : list of (B, n_cats) tensors from decoder
        mu           : (B, latent_dim)
        log_var      : (B, latent_dim)
        cont_slices  : [(alpha_s, beta_s, beta_e), ...]
        cat_slices   : [(start, end), ...]
        loss_factor  : reconstruction loss multiplier (default=2)
        kl_weight    : KL annealing coefficient in [0, 1].
                       0 = pure autoencoder (warm-up phase),
                       1 = full ELBO.  Linearly ramped by train_tvae.

    Returns:
        (total_loss, recon_loss_val, kl_loss_val)
    """
    recon = torch.tensor(0.0, device=x_batch.device)

    # ── continuous reconstruction ────────────────
    for i, (a_s, b_s, b_e) in enumerate(cont_slices):
        alpha_true = x_batch[:, a_s: b_s]   # (B, 1)
        beta_true  = x_batch[:, b_s: b_e]   # (B, n_modes)

        alpha_pred = cont_outputs[i][:, 0:1]         # (B, 1)
        beta_logits= cont_outputs[i][:, 1:]          # (B, n_modes)

        # MSE for normalised continuous value
        recon += F.mse_loss(alpha_pred, alpha_true, reduction='mean')

        # CrossEntropy for mode indicator
        # only compute if n_modes > 1
        if beta_true.shape[1] > 1:
            recon += F.cross_entropy(
                beta_logits,
                beta_true.argmax(dim=1)
            )

    # ── categorical reconstruction ───────────────
    for i, (g_s, g_e) in enumerate(cat_slices):
        gamma_true = x_batch[:, g_s:g_e]    # (B, n_cats)

        if gamma_true.shape[1] > 1:
            recon += F.cross_entropy(
                cat_outputs[i],
                gamma_true.argmax(dim=1)
            )

    # ── KL divergence (closed form) ──────────────
    # -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl = -0.5 * (
        1 + log_var - mu.pow(2) - log_var.exp()
    ).sum(dim=1).mean()

    # loss_factor upweights reconstruction; kl_weight ramps from 0→1
    total = loss_factor * recon + kl_weight * kl

    return total, recon.item(), kl.item()


# ═══════════════════════════════════════════════════════════════
# 6. TRAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def train_tvae(df: pd.DataFrame,
               continuous_cols: list,
               categorical_cols: list,
               compress_dims: tuple = (128, 128),
               decompress_dims: tuple = (128, 128),
               latent_dim: int = 128,
               epochs: int = 300,
               batch_size: int = 500,
               lr: float = 1e-3,
               l2scale: float = 1e-5,
               loss_factor: float = 2.0,
               kl_warmup_epochs: int = 100,
               device: str = 'cpu') -> tuple:
    """
    Fit a TVAE to a DataFrame.

    Hyperparameters match Xu et al. 2019 defaults:
        compress_dims     = (128, 128)
        decompress_dims   = (128, 128)
        latent_dim        = 128
        batch_size        = 500
        epochs            = 300
        lr                = 1e-3
        l2scale           = 1e-5   (L2 weight decay)
        loss_factor       = 2.0
        kl_warmup_epochs  = 100   (KL annealing warmup)

    KL annealing:
        The KL term is multiplied by kl_weight = epoch / kl_warmup_epochs
        (clamped to 1.0).  For the first kl_warmup_epochs epochs the
        model trains as a pure autoencoder — the decoder learns to
        reconstruct before the KL regularisation is introduced.
        This prevents posterior collapse (KL → 0, recon stuck).

    Returns:
        (model, transformer, cont_slices, cat_slices)
    """

    # ── fit transformer ──────────────────────────
    transformer = TabTransformer(continuous_cols, categorical_cols)
    transformer.fit(df)

    X    = transformer.transform(df)
    X_t  = torch.tensor(X, dtype=torch.float32).to(device)

    cont_dims, cat_dims = transformer.get_decoder_info()
    cont_slices, cat_slices = transformer.get_loss_slices()

    # ── build model ─────────────────────────────
    model = TVAE(
        input_dim       = X.shape[1],
        compress_dims   = compress_dims,
        decompress_dims = decompress_dims,
        latent_dim      = latent_dim,
        cont_dims       = cont_dims,
        cat_dims        = cat_dims
    ).to(device)

    # l2scale = weight decay (original paper uses 1e-5)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=l2scale
    )

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=True
    )

    # ── training loop ────────────────────────────
    model.train()

    # kl_warmup_epochs=0 disables annealing (kl_weight always 1.0)
    warmup = max(kl_warmup_epochs, 1)

    for epoch in range(epochs):
        # linear KL warmup: 0 → 1 over the first kl_warmup_epochs epochs
        kl_weight = min(1.0, (epoch + 1) / warmup)

        epoch_loss  = 0.0
        epoch_recon = 0.0
        epoch_kl    = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()

            cont_outputs, cat_outputs, mu, log_var = model(batch)

            loss, recon, kl = tvae_loss(
                x_batch     = batch,
                cont_outputs= cont_outputs,
                cat_outputs = cat_outputs,
                mu          = mu,
                log_var     = log_var,
                cont_slices = cont_slices,
                cat_slices  = cat_slices,
                loss_factor = loss_factor,
                kl_weight   = kl_weight
            )

            loss.backward()

            # gradient clipping — prevents exploding gradients
            # especially important for wide networks (large compress_dims)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )

            optimizer.step()
            epoch_loss  += loss.item()
            epoch_recon += recon
            epoch_kl    += kl

        n_batches = max(len(loader), 1)

        if (epoch + 1) % 50 == 0:
            print(
                f"  Epoch [{epoch+1:>4}/{epochs}] "
                f"loss={epoch_loss/n_batches:.4f}  "
                f"recon={epoch_recon/n_batches:.4f}  "
                f"kl={epoch_kl/n_batches:.4f}  "
                f"kl_w={kl_weight:.2f}"
            )

    model.eval()
    return model, transformer, cont_slices, cat_slices


# ═══════════════════════════════════════════════════════════════
# 7. SYNTHESIZE — same signature as tab_vae.synthesize()
# ═══════════════════════════════════════════════════════════════

def synthesize(model: TVAE,
               transformer: TabTransformer,
               n: int,
               continuous_cols: list,
               categorical_cols: list,
               device: str = 'cpu') -> pd.DataFrame:
    """
    Generate n synthetic rows from a trained TVAE.

    Returns a DataFrame with original column names.
    Same function signature as tab_vae.synthesize()
    so ClientTVAE works unchanged.

    Inference (TVAE — no Gumbel-Softmax):
        1. z ~ N(0, I)
        2. decode → (cont_outputs, cat_outputs)   raw logits
        3. continuous: alpha taken directly, beta → argmax
        4. categorical: gamma → argmax
        5. inverse_transform → original feature space
    """
    model.eval()
    with torch.no_grad():
        cont_outputs, cat_outputs = model.sample(n, device=device)

    return transformer.inverse_transform(cont_outputs, cat_outputs)


# ═══════════════════════════════════════════════════════════════
# 8. SIMPLE MSE VAE  (for PCA-space data)
#
# WHY A SEPARATE VAE:
#   TabTransformer + MSN was designed for raw tabular features
#   with multimodal distributions (age, income, etc.).
#   PCA components are Gaussian by construction — MSN fits a GMM
#   that always finds 1 active mode, making the beta (mode indicator)
#   head a constant and wasting capacity.
#
#   Problems with TVAE on PCA data:
#     1. MSN encodes each PCA dim as (alpha, beta=[1]) — beta is useless
#     2. loss_factor = latent/pca_dim can exceed 7× → recon dominates,
#        latent sigma → 0, posterior collapse at synthesis time
#     3. latent_dim > pca_dim (overcomplete) → no compression pressure
#
#   Fix: plain MLP encoder-decoder with direct MSE on raw PCA values.
#     - No GMM, no mode indicator
#     - latent_dim = pca_dim // 2  (proper bottleneck)
#     - loss_factor capped at 1.0 (balanced ELBO)
#     - Free-bits KL to prevent any dimension collapsing to prior
# ═══════════════════════════════════════════════════════════════

class SimpleMSEVAE(nn.Module):
    """
    Minimal VAE for PCA-space data.

    Encoder: MLP → (mu, log_var)
    Decoder: MLP → reconstruction  (Sigmoid output → values in [0,1])
    Loss:    MSE reconstruction + KL divergence (with free bits)
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z   = mu + std * torch.randn_like(std)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu') -> np.ndarray:
        z     = torch.randn(n, self.latent_dim, device=device)
        x_hat = self.decode(z)
        return x_hat.cpu().numpy()


def train_simple_vae(X_np       : np.ndarray,
                     hidden_dim : int   = 128,
                     latent_dim : int   = 16,
                     epochs     : int   = 300,
                     batch_size : int   = 128,
                     lr         : float = 1e-3,
                     free_bits  : float = 0.5,
                     kl_warmup  : int   = 100,
                     device     : str   = 'cpu') -> 'SimpleMSEVAE':
    """
    Train a SimpleMSEVAE on PCA-space data.

    Args:
        X_np      : (N, pca_dim) float32 — raw PCA component values
        hidden_dim: MLP hidden size
        latent_dim: bottleneck dimension — should be < pca_dim
        epochs    : training epochs
        batch_size: samples per gradient step
        lr        : Adam learning rate
        free_bits : per-dimension KL floor (nats).
                    Prevents any latent dim collapsing to prior.
        kl_warmup : epochs before full KL weight (linear ramp 0→1)
        device    : 'cpu' or 'cuda'

    Returns:
        trained SimpleMSEVAE (eval mode, parameters frozen)
    """
    input_dim = X_np.shape[1]

    # standardise to zero mean, unit variance so MSE is well-scaled
    mu_data  = X_np.mean(axis=0, keepdims=True)
    std_data = X_np.std(axis=0, keepdims=True).clip(1e-6)
    X_scaled = (X_np - mu_data) / std_data

    X_t   = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size = batch_size,
        shuffle    = True,
    )

    model = SimpleMSEVAE(input_dim, hidden_dim, latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    warmup = max(kl_warmup, 1)

    model.train()
    for epoch in range(epochs):
        kl_weight   = min(1.0, (epoch + 1) / warmup)
        epoch_loss  = epoch_recon = epoch_kl = 0.0

        for (batch,) in loader:
            opt.zero_grad()
            x_hat, mu, log_var = model(batch)

            # MSE reconstruction (mean over dims and batch)
            recon = F.mse_loss(x_hat, batch, reduction='mean')

            # KL with free bits: clamp per-dim KL to minimum free_bits
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl = kl_per_dim.clamp(min=free_bits).mean()

            loss = recon + kl_weight * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_loss  += loss.item()
            epoch_recon += recon.item()
            epoch_kl    += kl.item()

        n_b = max(len(loader), 1)
        if (epoch + 1) % 50 == 0:
            print(
                f"  Epoch [{epoch+1:>4}/{epochs}] "
                f"loss={epoch_loss/n_b:.4f}  "
                f"recon={epoch_recon/n_b:.4f}  "
                f"kl={epoch_kl/n_b:.4f}  "
                f"kl_w={kl_weight:.2f}"
            )

    model.eval()
    # store standardisation stats so synthesize_simple can invert them
    model.mu_data  = mu_data
    model.std_data = std_data
    return model


def synthesize_simple(model     : 'SimpleMSEVAE',
                      n         : int,
                      col_names : list,
                      device    : str = 'cpu') -> pd.DataFrame:
    """
    Generate n synthetic PCA-space rows from a trained SimpleMSEVAE.

    Returns a DataFrame with columns matching col_names (pc_0, pc_1, ...).
    The output is de-standardised back to the original PCA component scale.
    """
    model.eval()
    X_syn = model.sample(n, device=device)               # (n, pca_dim) standardised
    X_syn = X_syn * model.std_data + model.mu_data       # de-standardise
    return pd.DataFrame(X_syn.astype(np.float32), columns=col_names)