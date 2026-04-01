# rvae.py
#
# Recurrent Variational Autoencoder (RVAE) for sequential network traffic windows.
#
# Key differences from TVAE:
#   - No PCA preprocessing    — operates directly on (window_size, n_raw_features)
#   - No TabTransformer       — MSE on min-max normalised [0,1] features
#   - GRU encoder / decoder   — explicitly models temporal structure in windows
#   - Sigmoid output          — keeps synthetic values in [0, 1] matching input
#
# Anti-posterior-collapse measures (all three are active):
#   1. Free bits (Kingma et al. 2016, arXiv:1606.04934)
#      Per-dimension KL is clamped to a minimum of `free_bits` nats.
#      Dimensions below the threshold produce zero KL gradient, so the
#      encoder cannot satisfy the KL term by setting log_var → -∞.
#      This directly prevents KL = 0.
#
#   2. Cyclical KL annealing (Fu et al. 2019, arXiv:1903.10145)
#      kl_weight cycles 0 → 1 → 0 → 1 … (n_cycles times).
#      Each cycle: ramp up for 50% of cycle, hold at max for 50%.
#      The periodic resets let reconstruction recover first, then
#      KL pressure forces the latent code to be used.
#
#   3. Denoising (noise_std > 0)
#      Gaussian noise added to encoder input; decoder reconstructs
#      the CLEAN target. Forces z to carry sample-specific signal
#      rather than the dataset mean.
#
# Architecture:
#   Encoder : GRU(n_features → hidden_dim) → last hidden → fc_mu, fc_var
#   Decoder : z → fc_h0 (init state) + fc_input (repeated input) → GRU → fc_out → sigmoid

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1. ENCODER
# ═══════════════════════════════════════════════════════════════

class REncoder(nn.Module):
    """
    GRU encoder: (B, T, F) → (mu, log_var)

    Processes each window as a sequence of T time steps with F features.
    The final GRU hidden state is mapped to the latent Gaussian parameters.
    """

    def __init__(self, n_features: int, hidden_dim: int,
                 latent_dim: int, n_layers: int = 1):
        super().__init__()
        self.gru    = nn.GRU(n_features, hidden_dim, n_layers,
                             batch_first=True)
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # ── KEY FIX: non-zero initial log_var ───────────
        # Default PyTorch init → weights near 0 → log_var≈0 → KL=0
        # and grad(KL)=0 at that point — encoder can never escape.
        # bias=-1 → initial log_var≈-1 → KL_per_dim≈0.18 nats
        # → non-zero gradient from the very first step.
        nn.init.constant_(self.fc_var.bias, -1.0)

    def forward(self, x: torch.Tensor):
        # x : (B, T, F)
        _, h = self.gru(x)     # h : (n_layers, B, hidden_dim)
        h    = h[-1]           # last layer : (B, hidden_dim)
        return self.fc_mu(h), self.fc_var(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)


# ═══════════════════════════════════════════════════════════════
# 2. DECODER
# ═══════════════════════════════════════════════════════════════

class RDecoder(nn.Module):
    """
    GRU decoder: z → (B, T, F) in [0, 1]

    z is used in two ways to prevent the GRU from ignoring the latent code:
      1. fc_h0   → initial GRU hidden state  (structural information)
      2. fc_input → input at every time step  (repeated T times)

    Output is passed through sigmoid to keep values in [0, 1],
    matching the min-max normalised input.
    """

    def __init__(self, latent_dim: int, hidden_dim: int,
                 n_features: int, window_size: int, n_layers: int = 1):
        super().__init__()
        self.window_size = window_size
        self.n_layers    = n_layers
        self.hidden_dim  = hidden_dim

        self.fc_h0    = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_input = nn.Linear(latent_dim, n_features)
        self.gru      = nn.GRU(n_features, hidden_dim, n_layers,
                               batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_features)

    def forward(self, z: torch.Tensor):
        B   = z.size(0)
        h0  = self.fc_h0(z).view(self.n_layers, B, self.hidden_dim)
        inp = self.fc_input(z).unsqueeze(1).expand(-1, self.window_size, -1)
        out, _ = self.gru(inp, h0)               # (B, T, hidden_dim)
        return torch.sigmoid(self.fc_out(out))   # (B, T, F) in [0, 1]


# ═══════════════════════════════════════════════════════════════
# 3. RVAE — Full Model
# ═══════════════════════════════════════════════════════════════

class RVAE(nn.Module):
    """
    Recurrent Variational Autoencoder.

    Training:   encode (B,T,F) → reparameterize → decode → MSE + KL loss
    Inference:  z ~ N(0,I) → decode → (B,T,F) in [0,1]
    """

    def __init__(self, n_features: int, window_size: int,
                 hidden_dim: int = 128, latent_dim: int = 64,
                 n_layers: int = 1):
        super().__init__()
        self.n_features  = n_features
        self.window_size = window_size
        self.latent_dim  = latent_dim

        self.encoder = REncoder(n_features, hidden_dim, latent_dim, n_layers)
        self.decoder = RDecoder(latent_dim, hidden_dim, n_features,
                                window_size, n_layers)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z           = REncoder.reparameterize(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu',
               temperature: float = 1.0) -> torch.Tensor:
        """Sample n windows. temperature scales the prior std. Returns (n, T, F)."""
        z = temperature * torch.randn(n, self.latent_dim).to(device)
        return self.decoder(z)


# ═══════════════════════════════════════════════════════════════
# 4. LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════

def rvae_loss(recon: torch.Tensor,
              target: torch.Tensor,
              mu: torch.Tensor,
              log_var: torch.Tensor,
              kl_weight: float = 1.0,
              loss_factor: float = 1.0,
              free_bits: float = 0.5) -> tuple:
    """
    RVAE ELBO loss with free bits, KL annealing and loss balancing.

    recon_loss : MSE averaged over (B, T, F)  → O(1)
    kl_loss    : free-bits KL, summed over latent_dim → O(latent_dim)

    ── Loss balancing ───────────────────────────────────────────────
    Without loss_factor:
        recon ≈ 0.005   (mean over T×F features, sparse data)
        KL    ≈ 0.18 × latent_dim  (sum over latent dims)
    KL dominates → encoder drives KL→0 immediately.
    Fix: loss_factor = T×F converts recon to a feature-sum,
    making it commensurable with KL.

    ── Free bits (Kingma et al. 2016, arXiv:1606.04934) ─────────────
    Standard KL = -0.5 Σ(1 + log_var - mu² - exp(log_var)).sum(dim=1).mean()
    This can always be minimised to 0 by setting log_var → -∞, mu → 0.

    Free-bits KL:
        kl_per_dim = -0.5 * (1 + log_var - mu² - exp(log_var))  (B, D)
        kl_per_dim_avg = kl_per_dim.mean(dim=0)                  (D,)
        kl_loss = clamp(kl_per_dim_avg, min=free_bits).sum()

    Each latent dimension must carry at least free_bits nats of
    information.  Below the threshold the gradient is ZERO — the
    encoder cannot satisfy the regulariser by collapsing that dim.
    Above the threshold, normal VAE gradient flows.

    Typical values: free_bits = 0.5–2.0 nats/dim.

    Reference (architecture):
        GRU encoder/decoder — Cho et al. (2014), EMNLP
        Sequence VAE        — Bowman et al. (2016), EMNLP, arXiv:1511.06349
        Free bits           — Kingma et al. (2016), arXiv:1606.04934
        Cyclical annealing  — Fu et al. (2019), arXiv:1903.10145
    """
    recon_loss = F.mse_loss(recon, target, reduction='mean')

    # ── free-bits KL ─────────────────────────────────────────────
    # kl_elementwise : (B, latent_dim)  — per-sample per-dim KL
    kl_elementwise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    # average over batch so the threshold is per-dim, not per-sample
    kl_per_dim = kl_elementwise.mean(dim=0)          # (latent_dim,)
    # clamp: dims below free_bits contribute 0 gradient
    kl_loss = torch.clamp(kl_per_dim, min=free_bits).sum()

    total = loss_factor * recon_loss + kl_weight * kl_loss
    return total, recon_loss.item(), kl_loss.item()


# ═══════════════════════════════════════════════════════════════
# 5. TRAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def _cyclical_kl_weight(epoch: int, n_epochs: int,
                         n_cycles: int = 4,
                         min_w: float = 0.0,
                         max_w: float = 1.0) -> float:
    """
    Cyclical KL annealing schedule (Fu et al. 2019, arXiv:1903.10145).

    Divides training into n_cycles equal cycles.  Within each cycle:
      - First 50%: linear ramp from min_w → max_w
      - Last  50%: constant at max_w

    Rationale: periodic resets let the decoder re-learn reconstruction
    from scratch each cycle, then KL pressure forces the latent code
    to be used again.  This breaks the feedback loop where a powerful
    decoder ignores z (the root cause of posterior collapse in RNNs).
    """
    cycle_len = max(n_epochs / n_cycles, 1)
    pos       = (epoch % cycle_len) / cycle_len   # 0 → 1 within cycle
    ramp      = min(1.0, pos * 2.0)               # ramp in first 50%
    return min_w + (max_w - min_w) * ramp


def train_rvae(X_np: np.ndarray,
               window_size: int,
               n_raw_features: int,
               hidden_dim: int = 128,
               latent_dim: int = 64,
               n_layers: int = 1,
               epochs: int = 300,
               batch_size: int = 256,
               lr: float = 1e-3,
               kl_warmup_epochs: int = 100,   # kept for API compatibility
               noise_std: float = 0.10,
               free_bits: float = 1.0,
               n_cycles: int = 4,
               device: str = 'cpu') -> RVAE:
    """
    Train an RVAE on flat windows of shape (N, window_size * n_raw_features).

    Three anti-posterior-collapse mechanisms (all active by default):

    1. Free bits (free_bits > 0):
       Each latent dimension is forced to carry ≥ free_bits nats.
       Below the threshold the encoder gradient is zero — it cannot
       collapse by minimising KL.  Recommended: 0.5–2.0 nats/dim.

    2. Cyclical KL annealing (n_cycles > 1):
       kl_weight cycles 0 → 1 → 0 → 1 … n_cycles times.
       Each cycle: ramp for first 50%, hold for last 50%.
       The periodic resets let the decoder re-learn reconstruction
       each cycle, after which KL pressure forces latent code use.

    3. Denoising (noise_std > 0):
       Gaussian noise added to encoder input; decoder must
       reconstruct the CLEAN target from a noisy representation,
       forcing z to carry sample-specific signal.

    Returns a trained, eval-mode RVAE.
    """
    # reshape flat windows → sequences
    X   = X_np.reshape(-1, window_size, n_raw_features).astype(np.float32)
    X_t = torch.tensor(X).to(device)

    model     = RVAE(n_raw_features, window_size,
                     hidden_dim, latent_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=1e-5)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=True
    )

    model.train()

    # ── loss balancing ────────────────────────────────────────────
    # recon (mean over T×F) vs KL (sum over latent_dim):
    # loss_factor = T×F converts recon mean → equivalent feature-sum,
    # making both terms O(10-50) instead of O(0.005) vs O(10).
    loss_factor = float(window_size * n_raw_features)

    for epoch in range(epochs):
        # ── cyclical KL weight ────────────────────────────────────
        kl_weight   = _cyclical_kl_weight(epoch, epochs,
                                          n_cycles=n_cycles,
                                          min_w=0.0, max_w=1.0)
        epoch_loss  = 0.0
        epoch_recon = 0.0
        epoch_kl    = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()

            # ── denoising ─────────────────────────────────────────
            if noise_std > 0:
                noisy = batch + noise_std * torch.randn_like(batch)
                noisy = torch.clamp(noisy, 0.0, 1.0)
            else:
                noisy = batch

            recon, mu, log_var = model(noisy)
            loss, recon_val, kl_val = rvae_loss(
                recon, batch, mu, log_var,
                kl_weight   = kl_weight,
                loss_factor = loss_factor,
                free_bits   = free_bits,      # ← free bits applied here
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_recon += recon_val
            epoch_kl    += kl_val

        n_batches = max(len(loader), 1)
        if (epoch + 1) % 50 == 0:
            print(
                f"  Epoch [{epoch+1:>4}/{epochs}] "
                f"loss={epoch_loss/n_batches:.4f}  "
                f"recon={epoch_recon/n_batches:.4f}  "
                f"kl={epoch_kl/n_batches:.4f}  "
                f"kl_w={kl_weight:.2f}  "
                f"free_bits={free_bits:.1f}"
            )

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# 6. LSTM WITH ATTENTION ENCODER / DECODER
# ═══════════════════════════════════════════════════════════════
#
# Encoder: LSTM → all hidden states (B, T, H)
#          → dot-product self-attention over T → context vector (B, H)
#          → fc_mu, fc_var → (mu, log_var)
#
# Decoder: z → fc_h0 (init state) + fc_input (repeated T times)
#          → LSTM → all hidden states (B, T, H)
#          → dot-product self-attention over T → attended (B, T, H)
#          → fc_out → sigmoid → (B, T, F)
#
# Why attention helps over plain GRU/LSTM:
#   Plain RNN encodes the whole window into the LAST hidden state —
#   early timesteps are compressed and may be lost.
#   Attention lets the encoder look back at ALL timesteps and weight
#   them by relevance, capturing long-range dependencies within a window.


class LSTMAttnEncoder(nn.Module):
    """
    LSTM encoder with dot-product self-attention over timesteps.

    (B, T, F) → LSTM → all hidden (B, T, H)
              → attention weights (B, T)
              → weighted sum context (B, H)
              → fc_mu, fc_var → (mu, log_var)
    """

    def __init__(self, n_features: int, hidden_dim: int,
                 latent_dim: int, n_layers: int = 1):
        super().__init__()
        self.lstm   = nn.LSTM(n_features, hidden_dim, n_layers,
                              batch_first=True)
        self.attn_w = nn.Linear(hidden_dim, 1)   # attention scoring
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.fc_var.bias, -1.0)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        out, _ = self.lstm(x)              # out: (B, T, H)

        # dot-product attention: score each timestep
        scores  = self.attn_w(out)         # (B, T, 1)
        weights = torch.softmax(scores, dim=1)   # (B, T, 1)
        context = (weights * out).sum(dim=1)     # (B, H)

        return self.fc_mu(context), self.fc_var(context)

    @staticmethod
    def reparameterize(mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)


class LSTMAttnDecoder(nn.Module):
    """
    LSTM decoder with dot-product self-attention over generated timesteps.

    z → fc_h0 (LSTM init state) + fc_input (repeated T times)
      → LSTM → all hidden (B, T, H)
      → self-attention over T → attended (B, T, H)
      → fc_out → sigmoid → (B, T, F)
    """

    def __init__(self, latent_dim: int, hidden_dim: int,
                 n_features: int, window_size: int, n_layers: int = 1):
        super().__init__()
        self.window_size = window_size
        self.n_layers    = n_layers
        self.hidden_dim  = hidden_dim

        self.fc_h0    = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_c0    = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_input = nn.Linear(latent_dim, n_features)
        self.lstm     = nn.LSTM(n_features, hidden_dim, n_layers,
                                batch_first=True)
        self.attn_w   = nn.Linear(hidden_dim, 1)
        self.fc_out   = nn.Linear(hidden_dim, n_features)

    def forward(self, z: torch.Tensor):
        B   = z.size(0)
        h0  = self.fc_h0(z).view(self.n_layers, B, self.hidden_dim)
        c0  = self.fc_c0(z).view(self.n_layers, B, self.hidden_dim)
        inp = self.fc_input(z).unsqueeze(1).expand(-1, self.window_size, -1)

        out, _ = self.lstm(inp, (h0, c0))          # (B, T, H)

        # self-attention over decoder timesteps
        scores  = self.attn_w(out)                  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)      # (B, T, 1)
        attended = weights * out                    # (B, T, H)

        return torch.sigmoid(self.fc_out(attended))  # (B, T, F)


class LSTMAttnVAE(nn.Module):
    """
    VAE with LSTM + attention encoder and decoder.

    Same API as RVAE — drop-in replacement.
    """

    def __init__(self, n_features: int, window_size: int,
                 hidden_dim: int = 128, latent_dim: int = 64,
                 n_layers: int = 1):
        super().__init__()
        self.n_features  = n_features
        self.window_size = window_size
        self.latent_dim  = latent_dim

        self.encoder = LSTMAttnEncoder(n_features, hidden_dim, latent_dim, n_layers)
        self.decoder = LSTMAttnDecoder(latent_dim, hidden_dim, n_features,
                                       window_size, n_layers)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z           = LSTMAttnEncoder.reparameterize(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu',
               temperature: float = 1.0) -> torch.Tensor:
        z = temperature * torch.randn(n, self.latent_dim).to(device)
        return self.decoder(z)


def train_lstm_attn_vae(X_np: np.ndarray,
                        window_size: int,
                        n_raw_features: int,
                        hidden_dim: int = 128,
                        latent_dim: int = 64,
                        n_layers: int = 1,
                        epochs: int = 300,
                        batch_size: int = 256,
                        lr: float = 1e-3,
                        noise_std: float = 0.10,
                        free_bits: float = 1.0,
                        n_cycles: int = 4,
                        device: str = 'cpu') -> LSTMAttnVAE:
    """Train a LSTMAttnVAE. Same signature as train_rvae."""
    X   = X_np.reshape(-1, window_size, n_raw_features).astype(np.float32)
    X_t = torch.tensor(X).to(device)

    model     = LSTMAttnVAE(n_raw_features, window_size,
                             hidden_dim, latent_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=1e-5)
    loader    = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True
    )
    loss_factor = float(window_size * n_raw_features)

    model.train()
    for epoch in range(epochs):
        kl_weight   = _cyclical_kl_weight(epoch, epochs, n_cycles=n_cycles)
        epoch_loss  = epoch_recon = epoch_kl = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()
            if noise_std > 0:
                noisy = (batch + noise_std * torch.randn_like(batch)).clamp(0, 1)
            else:
                noisy = batch

            recon, mu, log_var = model(noisy)
            loss, recon_val, kl_val = rvae_loss(
                recon, batch, mu, log_var,
                kl_weight=kl_weight, loss_factor=loss_factor, free_bits=free_bits
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss  += loss.item()
            epoch_recon += recon_val
            epoch_kl    += kl_val

        n_b = max(len(loader), 1)
        if (epoch + 1) % 50 == 0:
            print(
                f"  [LSTMAttn] Epoch [{epoch+1:>4}/{epochs}] "
                f"loss={epoch_loss/n_b:.4f}  recon={epoch_recon/n_b:.4f}  "
                f"kl={epoch_kl/n_b:.4f}  kl_w={kl_weight:.2f}"
            )

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# 7. TRANSFORMER-BASED VAE ENCODER / MLP DECODER
# ═══════════════════════════════════════════════════════════════
#
# Encoder: linear projection (F → d_model)
#          + sinusoidal positional encoding
#          + TransformerEncoder (multi-head self-attention + FFN)
#          → mean pool over T → fc_mu, fc_var
#
# Decoder: MLP only — z → Linear → LayerNorm → GELU
#                        → Linear → LayerNorm → GELU
#                        → Linear → reshape (B, T, F) → sigmoid
#
# Why MLP decoder (not TransformerDecoder):
#   The VAE is NOT autoregressive — z already encodes the full window.
#   A TransformerDecoder needs a target sequence to cross-attend to,
#   which doesn't exist at generation time without sequential sampling.
#   An MLP decoder maps z → full output in one shot, which is correct
#   for unconditional generation from a learned prior N(0,I).
#   The Transformer encoder still benefits from attention — it compresses
#   the input window more faithfully than a GRU by attending to all
#   T timestep pairs simultaneously.


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding added to input embeddings.
    Injects position information without learned parameters.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerEncoder_VAE(nn.Module):
    """
    Transformer encoder: (B, T, F) → (mu, log_var)

    Projects F → d_model, adds positional encoding,
    passes through N transformer encoder layers,
    pools over T by mean, maps to latent params.
    """

    def __init__(self, n_features: int, d_model: int, latent_dim: int,
                 n_heads: int = 4, n_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.fc_mu  = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)
        nn.init.constant_(self.fc_var.bias, -1.0)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        e = self.pos_enc(self.input_proj(x))   # (B, T, d_model)
        e = self.transformer(e)                # (B, T, d_model)
        pooled = e.mean(dim=1)                 # (B, d_model) — mean pooling over T
        return self.fc_mu(pooled), self.fc_var(pooled)

    @staticmethod
    def reparameterize(mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)


class MLPDecoder_VAE(nn.Module):
    """
    MLP decoder: z → (B, T, F)

    Maps the latent vector unconditionally to the full output window
    in one shot — correct for VAE generation from z ~ N(0, I).

    Architecture:
        z (B, latent_dim)
        → Linear(latent_dim, hidden) → LayerNorm → GELU
        → Linear(hidden, hidden)     → LayerNorm → GELU
        → Linear(hidden, T * F)
        → reshape (B, T, F)
        → sigmoid → [0, 1]

    LayerNorm + GELU stabilises training without dropout.
    """

    def __init__(self, latent_dim: int, hidden_dim: int,
                 n_features: int, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.n_features  = n_features
        out_dim          = window_size * n_features

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor):
        out = self.net(z)                                          # (B, T*F)
        return torch.sigmoid(out.view(-1, self.window_size,
                                      self.n_features))            # (B, T, F)


class TransformerVAE(nn.Module):
    """
    Transformer-based Variational Autoencoder.

    Encoder : TransformerEncoder — multi-head self-attention over T timesteps
              → mean pool → mu, log_var
    Decoder : MLP — z → full output window in one shot (no sequential decoding)

    Same API as RVAE — drop-in replacement.
    """

    def __init__(self, n_features: int, window_size: int,
                 d_model: int = 64, latent_dim: int = 32,
                 n_heads: int = 4, n_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_features  = n_features
        self.window_size = window_size
        self.latent_dim  = latent_dim

        self.encoder = TransformerEncoder_VAE(
            n_features, d_model, latent_dim,
            n_heads, n_layers, dim_feedforward, dropout
        )
        # MLP hidden_dim mirrors dim_feedforward for consistency
        self.decoder = MLPDecoder_VAE(
            latent_dim, dim_feedforward, n_features, window_size
        )

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z           = TransformerEncoder_VAE.reparameterize(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu',
               temperature: float = 1.0) -> torch.Tensor:
        z = temperature * torch.randn(n, self.latent_dim).to(device)
        return self.decoder(z)


def train_transformer_vae(X_np: np.ndarray,
                          window_size: int,
                          n_raw_features: int,
                          d_model: int = 64,
                          latent_dim: int = 32,
                          n_heads: int = 4,
                          n_layers: int = 2,
                          dim_feedforward: int = 256,
                          dropout: float = 0.1,
                          epochs: int = 300,
                          batch_size: int = 256,
                          lr: float = 1e-3,
                          noise_std: float = 0.10,
                          free_bits: float = 1.0,
                          n_cycles: int = 4,
                          device: str = 'cpu') -> TransformerVAE:
    """Train a TransformerVAE. d_model must be divisible by n_heads."""
    assert d_model % n_heads == 0, \
        f"d_model={d_model} must be divisible by n_heads={n_heads}"

    X   = X_np.reshape(-1, window_size, n_raw_features).astype(np.float32)
    X_t = torch.tensor(X).to(device)

    model     = TransformerVAE(n_raw_features, window_size, d_model, latent_dim,
                               n_heads, n_layers, dim_feedforward, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader    = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True
    )
    loss_factor = float(window_size * n_raw_features)

    model.train()
    for epoch in range(epochs):
        kl_weight   = _cyclical_kl_weight(epoch, epochs, n_cycles=n_cycles)
        epoch_loss  = epoch_recon = epoch_kl = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()
            if noise_std > 0:
                noisy = (batch + noise_std * torch.randn_like(batch)).clamp(0, 1)
            else:
                noisy = batch

            recon, mu, log_var = model(noisy)
            loss, recon_val, kl_val = rvae_loss(
                recon, batch, mu, log_var,
                kl_weight=kl_weight, loss_factor=loss_factor, free_bits=free_bits
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss  += loss.item()
            epoch_recon += recon_val
            epoch_kl    += kl_val

        n_b = max(len(loader), 1)
        if (epoch + 1) % 50 == 0:
            print(
                f"  [TransformerVAE] Epoch [{epoch+1:>4}/{epochs}] "
                f"loss={epoch_loss/n_b:.4f}  recon={epoch_recon/n_b:.4f}  "
                f"kl={epoch_kl/n_b:.4f}  kl_w={kl_weight:.2f}"
            )

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# 8. SYNTHESIZE
# ═══════════════════════════════════════════════════════════════

def synthesize(model: RVAE,
               n: int,
               window_size: int,
               n_raw_features: int,
               device: str = 'cpu',
               temperature: float = 1.0) -> np.ndarray:
    """
    Generate n synthetic flat windows.

    temperature > 1  → more diverse / spread-out samples
    temperature < 1  → samples closer to the learned mean
    temperature = 1  → standard prior N(0, I)  (default)

    Returns (n, window_size * n_raw_features) float32 array in [0, 1].
    This matches the format produced by utils.load_data so the output
    can be directly used as replay data for the LSTM classifier.
    """
    model.eval()
    with torch.no_grad():
        z       = temperature * torch.randn(n, model.latent_dim).to(device)
        samples = model.decoder(z)             # (n, T, F)
    return samples.cpu().numpy().reshape(n, window_size * n_raw_features)
