# time_vae.py
#
# Base TimeVAE — PyTorch implementation of:
#   Desai et al. (2021) "TimeVAE: A Variational Auto-Encoder for
#   Multivariate Time Series Generation"  arXiv:2111.08095
#
# Input tensor convention (matches utils.load_data output):
#   (B, T, F)   B=batch, T=window_size (time steps), F=n_raw_features
#
# SHORT-WINDOW NOTE (T < 5, e.g. T=3):
#   With very short windows Conv1D with kernel_size=3 spans the entire
#   window — it degenerates into a dense layer with padding artefacts.
#   When T <= 4 the architecture automatically uses a Dense-only path
#   (Flatten → MLP → latent) instead of Conv1D so the model is still
#   well-defined and trains stably.
#
# Public API:
#   train_time_vae(X_np, T, F, ...)  → TimeVAE model
#   synthesize_time_vae(model, n, device)  → np.ndarray (n, T, F)

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Threshold below which Conv1D is replaced by a Dense path
_CONV_MIN_T = 5


# ═══════════════════════════════════════════════════════════════
# 1.  ENCODER
# ═══════════════════════════════════════════════════════════════

class TimeVAEEncoder(nn.Module):
    """
    Encoder for (B, T, F) windows.

    When T >= _CONV_MIN_T:
        permute (B,T,F) → (B,F,T)
        Conv1D stack (channels=F → filters)
        Flatten → Dense → (mu, log_var)

    When T < _CONV_MIN_T (e.g. default window_size=3):
        Flatten (B, T*F)
        MLP (two hidden layers of size hidden_dim)
        Dense → (mu, log_var)

    Both paths produce identical output shapes so the rest of the
    model and training loop are unaffected.
    """

    def __init__(self, T: int, F: int, latent_dim: int,
                 filters: list[int] = (32, 64),
                 kernel_size: int = 3,
                 hidden_dim: int = 128):
        super().__init__()
        self.T          = T
        self.F          = F
        self.latent_dim = latent_dim
        self.use_conv   = (T >= _CONV_MIN_T)

        if self.use_conv:
            layers = []
            in_ch  = F
            for out_ch in filters:
                layers += [
                    nn.Conv1d(in_ch, out_ch,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2),
                    nn.ReLU(),
                ]
                in_ch = out_ch
            self.conv_stack = nn.Sequential(*layers)
            flat_dim = in_ch * T
        else:
            # Dense path for short windows
            flat_dim = T * F
            self.dense_stack = nn.Sequential(
                nn.Linear(flat_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            flat_dim = hidden_dim

        self.fc_mu  = nn.Linear(flat_dim, latent_dim)
        self.fc_var = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        if self.use_conv:
            h = x.permute(0, 2, 1)     # (B, F, T)
            h = self.conv_stack(h)      # (B, C_last, T)
            h = h.flatten(1)           # (B, C_last * T)
        else:
            h = x.flatten(1)           # (B, T*F)
            h = self.dense_stack(h)    # (B, hidden_dim)

        mu      = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var


# ═══════════════════════════════════════════════════════════════
# 2.  DECODER
# ═══════════════════════════════════════════════════════════════

class TimeVAEDecoder(nn.Module):
    """
    Decoder that mirrors the encoder path.

    Conv path  : Dense → Reshape (B, C_first, T) → ConvTranspose1D → TimeDist Linear → (B, T, F)
    Dense path : Dense → MLP → Dense → Reshape (B, T, F)
    """

    def __init__(self, T: int, F: int, latent_dim: int,
                 filters: list[int] = (64, 32),
                 kernel_size: int = 3,
                 hidden_dim: int = 128):
        super().__init__()
        self.T        = T
        self.F        = F
        self.use_conv = (T >= _CONV_MIN_T)

        if self.use_conv:
            first_ch = filters[0]
            self.fc_expand = nn.Linear(latent_dim, first_ch * T)

            layers = []
            in_ch  = first_ch
            for out_ch in filters[1:]:
                layers += [
                    nn.ConvTranspose1d(in_ch, out_ch,
                                       kernel_size=kernel_size,
                                       padding=kernel_size // 2),
                    nn.ReLU(),
                ]
                in_ch = out_ch
            self.deconv_stack = nn.Sequential(*layers)
            self.last_ch      = in_ch
            self.td_linear    = nn.Linear(in_ch, F)
        else:
            # Dense path
            self.dense_stack = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, T * F),
            )

    def forward(self, z: torch.Tensor):
        B = z.size(0)
        if self.use_conv:
            h   = self.fc_expand(z)               # (B, first_ch * T)
            h   = h.view(B, -1, self.T)            # (B, first_ch, T)
            h   = self.deconv_stack(h)             # (B, last_ch, T)
            h   = h.permute(0, 2, 1)              # (B, T, last_ch)
            out = self.td_linear(h)                # (B, T, F)
        else:
            h   = self.dense_stack(z)              # (B, T*F)
            out = h.view(B, self.T, self.F)        # (B, T, F)
        return out


# ═══════════════════════════════════════════════════════════════
# 3.  TIMEVAE — Full model
# ═══════════════════════════════════════════════════════════════

class TimeVAE(nn.Module):
    """
    Base TimeVAE.

    Training   : encode → reparameterize → decode → ELBO
    Generation : z ~ N(0,I) → decode → (B, T, F)
    """

    def __init__(self, T: int, F: int,
                 latent_dim: int       = 16,
                 enc_filters: list[int] = (32, 64),
                 kernel_size: int      = 3,
                 hidden_dim: int       = 128):
        super().__init__()
        self.T          = T
        self.F          = F
        self.latent_dim = latent_dim

        dec_filters = list(reversed(enc_filters))

        self.encoder = TimeVAEEncoder(
            T=T, F=F,
            latent_dim  = latent_dim,
            filters     = enc_filters,
            kernel_size = kernel_size,
            hidden_dim  = hidden_dim,
        )
        self.decoder = TimeVAEDecoder(
            T=T, F=F,
            latent_dim  = latent_dim,
            filters     = dec_filters,
            kernel_size = kernel_size,
            hidden_dim  = hidden_dim,
        )

    @staticmethod
    def _reparam(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z           = self._reparam(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu') -> np.ndarray:
        """Returns (n, T, F) float32."""
        z     = torch.randn(n, self.latent_dim, device=device)
        x_hat = self.decoder(z)
        return x_hat.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# 4.  ELBO LOSS
# ═══════════════════════════════════════════════════════════════

def timevae_loss(x           : torch.Tensor,
                 x_hat       : torch.Tensor,
                 mu          : torch.Tensor,
                 log_var     : torch.Tensor,
                 recon_weight: float = 1.0,
                 kl_weight   : float = 1.0,
                 free_bits   : float = 0.0,
                 ) -> tuple[torch.Tensor, float, float]:
    """
    ELBO = recon_weight * MSE(x_hat, x) + kl_weight * KL

    recon_weight is scaled by T*F internally so the balance between
    reconstruction and KL does not change with window size or feature
    count — only the user-supplied recon_weight scalar matters.
    """
    T_F = x.shape[1] * x.shape[2]   # total elements per sample

    # Mean over batch; sum over T and F so signal doesn't shrink with T
    recon = F.mse_loss(x_hat, x, reduction='sum') / x.shape[0]

    # KL: closed-form Gaussian
    kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())  # (B, m)
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl = kl_per_dim.sum(dim=1).mean()   # sum over m, mean over B

    # Scale recon_weight by 1/T_F so the effective weight is invariant to
    # window shape (matches the paper's normalised ELBO convention)
    total = (recon_weight / T_F) * recon + kl_weight * kl
    return total, recon.item() / T_F, kl.item()


# ═══════════════════════════════════════════════════════════════
# 5.  TRAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def train_time_vae(X_np        : np.ndarray,
                   T           : int,
                   F           : int,
                   latent_dim  : int   = 16,
                   enc_filters : tuple = (32, 64),
                   kernel_size : int   = 3,
                   hidden_dim  : int   = 128,
                   epochs      : int   = 300,
                   batch_size  : int   = 128,
                   lr          : float = 1e-3,
                   recon_weight: float = 1.0,
                   kl_warmup   : int   = 100,
                   free_bits   : float = 0.5,
                   device      : str   = 'cpu') -> TimeVAE:
    """
    Fit a Base TimeVAE to windowed time-series data.

    Args:
        X_np        : (N, T, F) float32 — normalised windows [0, 1]
        T           : time steps per window (window_size)
        F           : features per step (n_raw_features)
        latent_dim  : bottleneck dimension
        enc_filters : Conv1D channel counts per encoder layer
                      (ignored when T < _CONV_MIN_T — dense path used)
        kernel_size : temporal conv kernel size
        hidden_dim  : hidden layer width for the dense path (T < 5)
        epochs      : training epochs
        batch_size  : samples per gradient step
        lr          : Adam learning rate
        recon_weight: scalar on reconstruction term (paper: 0.5–3.5)
        kl_warmup   : epochs for linear KL ramp (0 = always full KL)
        free_bits   : per-dim KL floor; prevents latent collapse
        device      : 'cpu' or 'cuda'

    Returns:
        Trained TimeVAE (eval mode)
    """
    assert X_np.ndim == 3, (
        f"Expected (N, T, F), got shape {X_np.shape}. "
        f"Make sure you pass the 3-D tensor from the DataLoader, "
        f"not the flattened (N, T*F) version."
    )
    assert X_np.shape[1] == T and X_np.shape[2] == F, (
        f"Shape mismatch: X_np is {X_np.shape} but T={T}, F={F} declared."
    )

    if T < _CONV_MIN_T:
        print(
            f"  [TimeVAE] T={T} < {_CONV_MIN_T} — "
            f"using Dense path (no Conv1D)."
        )

    X_t    = torch.tensor(X_np, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size = batch_size,
        shuffle    = True,
    )

    model = TimeVAE(
        T           = T,
        F           = F,
        latent_dim  = latent_dim,
        enc_filters = list(enc_filters),
        kernel_size = kernel_size,
        hidden_dim  = hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=1e-5)
    warmup = max(kl_warmup, 1)

    model.train()
    for epoch in range(epochs):
        kl_weight = min(1.0, (epoch + 1) / warmup)

        epoch_loss = epoch_recon = epoch_kl = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()

            x_hat, mu, log_var = model(batch)

            loss, recon, kl = timevae_loss(
                x            = batch,
                x_hat        = x_hat,
                mu           = mu,
                log_var      = log_var,
                recon_weight = recon_weight,
                kl_weight    = kl_weight,
                free_bits    = free_bits,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_recon += recon
            epoch_kl    += kl

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
    return model


# ═══════════════════════════════════════════════════════════════
# 6.  SYNTHESIZE FUNCTION
# ═══════════════════════════════════════════════════════════════

def synthesize_time_vae(model  : TimeVAE,
                        n      : int,
                        device : str = 'cpu') -> np.ndarray:
    """
    Generate n synthetic windows from a trained TimeVAE.

    Returns np.ndarray of shape (n, T, F), float32, clipped to [0, 1].
    """
    model.eval()
    X_syn = model.sample(n, device=device)   # (n, T, F)
    return np.clip(X_syn, 0.0, 1.0).astype(np.float32)
