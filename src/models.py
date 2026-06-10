import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class LSTMClassifier(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, num_layers=1, fc_hidden_dim=64, head_dropout: float = 0.0):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))   # out: (B, T, H)

        # Take last time step
        feat = out[:, -1, :]                        # (B, H)

        # Two-layer head
        feat = F.relu(self.fc1(feat))
        feat = self.dropout(feat)
        logits = self.fc2(feat)                     # raw logits (B, C)

        return logits, (h_n, c_n)
        

class LSTMModelWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0, bidirectional=False):
        super(LSTMModelWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.attn_fc = nn.Linear(hidden_size * self.num_directions, 1)
        self.output_fc = nn.Linear(hidden_size * self.num_directions, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def attention_net(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim*num_directions]
        attn_weights = self.attn_fc(lstm_output)  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # [batch_size, seq_len, 1]
        weighted_output = lstm_output * attn_weights  # [batch_size, seq_len, hidden_dim*num_directions]
        context_vector = weighted_output.sum(dim=1)  # [batch_size, hidden_dim*num_directions]
        return context_vector

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)

        lstm_out, _ = self.lstm(x, (h0, c0))  # [batch_size, seq_len, hidden_dim*num_directions]

        # Apply attention mechanism
        context = self.attention_net(lstm_out)

        # Pass through output layer
        output = self.output_fc(context)
        return output, context



import torch
import torch.nn as nn
import torch.nn.functional as F


class CTVAE(nn.Module):
    """
    Conditional Sequential VAE for network traffic windows.

    Architecture
    ────────────
    Encoder:
      y → embedding (embed_dim,)
      x (B, T, F) + tiled embedding → (B, T, F+embed_dim)
      GRU → last hidden → fc_mu, fc_logvar → z (B, latent_dim)

    Decoder:
      concat(z, embed(y)) → fc_h0 → initial GRU hidden state
                          → fc_inp → repeated T times as GRU input
      GRU → fc_out → sigmoid → (B, T, F) in [0, 1]

    Why conditioning matters
    ────────────────────────
    Without class conditioning the VAE must map both benign and
    attack windows to the same prior N(0,I).  The latent space
    ends up blurred and synthesised samples carry no class signal.
    With conditioning, each class gets its own region of latent
    space, so generate(class=1) reliably yields attack windows.

    Anti-posterior-collapse measures (same as RVAE)
    ────────────────────────────────────────────────
    1. Free bits     — per-dim KL clamped to ≥ free_bits nats
    2. Cyclical KL   — kl_weight oscillates 0→1 over n_cycles
    3. Denoising     — Gaussian noise on encoder input
    4. fc_logvar bias = -1 — non-zero initial gradient from step 0
    """

    def __init__(self,
                 n_features  : int,
                 window_size : int,
                 num_classes : int  = 2,
                 hidden_dim  : int  = 128,
                 latent_dim  : int  = 16,
                 embed_dim   : int  = 8,
                 n_layers    : int  = 1):
        super().__init__()

        self.n_features  = n_features
        self.window_size = window_size
        self.num_classes = num_classes
        self.latent_dim  = latent_dim
        self.n_layers    = n_layers
        self.hidden_dim  = hidden_dim

        # ── class embedding (shared encoder + decoder) ───────────
        self.embed = nn.Embedding(num_classes, embed_dim)

        # ── encoder ──────────────────────────────────────────────
        # input at each step: [feature_values | class_embedding]
        self.enc_gru    = nn.GRU(n_features + embed_dim, hidden_dim,
                                 n_layers, batch_first=True)
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar  = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.fc_logvar.bias, -1.0)  # avoid KL=0 at init

        # ── decoder ──────────────────────────────────────────────
        # z and class embedding are concatenated before decoding
        dec_in = latent_dim + embed_dim
        self.fc_h0    = nn.Linear(dec_in, n_layers * hidden_dim)
        self.fc_inp   = nn.Linear(dec_in, n_features)
        self.dec_gru  = nn.GRU(n_features, hidden_dim, n_layers,
                                batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_features)

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _encode(self, x, y):
        """x: (B,T,F)  y: (B,)  →  mu, logvar: (B, latent_dim)"""
        e   = self.embed(y).unsqueeze(1).expand(-1, self.window_size, -1)  # (B,T,E)
        inp = torch.cat([x, e], dim=-1)                                    # (B,T,F+E)
        _, h = self.enc_gru(inp)                                            # h: (L,B,H)
        h    = h[-1]                                                        # (B,H)
        return self.fc_mu(h), self.fc_logvar(h)

    def _decode(self, z, y):
        """z: (B, latent_dim)  y: (B,)  →  recon: (B,T,F)"""
        B   = z.size(0)
        e   = self.embed(y)                                                 # (B,E)
        dec = torch.cat([z, e], dim=-1)                                     # (B, latent+E)
        h0  = self.fc_h0(dec).view(self.n_layers, B, self.hidden_dim)       # (L,B,H)
        inp = self.fc_inp(dec).unsqueeze(1).expand(-1, self.window_size, -1) # (B,T,F)
        out, _ = self.dec_gru(inp, h0)                                      # (B,T,H)
        return torch.sigmoid(self.fc_out(out))                              # (B,T,F)

    # ── forward ──────────────────────────────────────────────────
    def forward(self, x, y):
        """
        x : (B, T, F)  normalised windows in [0,1]
        y : (B,)       integer class labels
        returns: recon (B,T,F), mu (B,D), logvar (B,D)
        """
        mu, logvar = self._encode(x, y)
        z          = self.reparameterize(mu, logvar)
        recon      = self._decode(z, y)
        return recon, mu, logvar

    # ── sampling ─────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, n, class_label, device='cpu', temperature=1.0):
        """
        Generate n windows conditioned on a single integer class_label.

        temperature > 1 → more diverse samples
        temperature < 1 → samples closer to the learned class mean
        Returns (n, T, F) float32 numpy array in [0,1].
        """
        self.eval()
        z = temperature * torch.randn(n, self.latent_dim, device=device)
        y = torch.full((n,), class_label, dtype=torch.long, device=device)
        out = self._decode(z, y)                          # (n, T, F)
        return out.cpu().numpy()


# ── LOSS ─────────────────────────────────────────────────────────────────────

def ctvae_loss(recon, target, mu, logvar,
               kl_weight=1.0, loss_factor=1.0, free_bits=0.5):
    """
    ELBO loss with free-bits KL and loss balancing.

    recon_loss : MSE averaged over (B, T, F)   — O(1)
    kl_loss    : free-bits KL summed over D    — O(D)
    loss_factor scales recon to be commensurable with kl_loss.
    """
    recon_loss     = F.mse_loss(recon, target, reduction='mean')
    kl_elem        = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B,D)
    kl_per_dim     = kl_elem.mean(dim=0)                              # (D,)
    kl_loss        = torch.clamp(kl_per_dim, min=free_bits).sum()
    total          = loss_factor * recon_loss + kl_weight * kl_loss
    return total, recon_loss.item(), kl_loss.item()


# ── TRAINING FUNCTION ─────────────────────────────────────────────────────────

def _cyclical_kl_weight(epoch, n_epochs, n_cycles=4, min_w=0.0, max_w=1.0):
    cycle_len = max(n_epochs / n_cycles, 1)
    pos       = (epoch % cycle_len) / cycle_len
    ramp      = min(1.0, pos * 2.0)
    return min_w + (max_w - min_w) * ramp


def train_ctvae(X_np, y_np,
                window_size, n_raw_features, num_classes=2,
                hidden_dim=128, latent_dim=16, embed_dim=8,
                n_layers=1, epochs=300, batch_size=256,
                lr=1e-3, noise_std=0.10, free_bits=1.0,
                n_cycles=4, device='cpu'):
    """
    Train a CTVAE on real windowed data.

    X_np : (N, window_size * n_raw_features)  float32, values in [0,1]
    y_np : (N,)  integer class labels

    Returns a trained, eval-mode CTVAE.
    """
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    # flat windows → sequences (N, T, F)
    X = X_np.reshape(-1, window_size, n_raw_features).astype('float32')
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y_np.astype('int64'), device=device)

    model     = CTVAE(n_raw_features, window_size, num_classes,
                      hidden_dim, latent_dim, embed_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size, shuffle=True
    )

    # loss_factor: scale recon (mean over T×F) up to match KL (sum over D)
    loss_factor = float(window_size * n_raw_features)

    model.train()
    for epoch in range(epochs):
        kl_w        = _cyclical_kl_weight(epoch, epochs, n_cycles)
        epoch_loss  = epoch_recon = epoch_kl = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()

            # denoising: add noise to encoder input, reconstruct clean target
            if noise_std > 0:
                noisy = torch.clamp(xb + noise_std * torch.randn_like(xb), 0.0, 1.0)
            else:
                noisy = xb

            recon, mu, logvar = model(noisy, yb)
            loss, rv, kv = ctvae_loss(recon, xb, mu, logvar,
                                      kl_weight=kl_w,
                                      loss_factor=loss_factor,
                                      free_bits=free_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_recon += rv
            epoch_kl    += kv

        nb = max(len(loader), 1)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1:>4}/{epochs}] "
                  f"loss={epoch_loss/nb:.4f}  "
                  f"recon={epoch_recon/nb:.4f}  "
                  f"kl={epoch_kl/nb:.4f}  "
                  f"kl_w={kl_w:.2f}")

    model.eval()
    return model


# ── ZI-RVAE: Zero-Inflated Recurrent VAE for EFL generator training ──────────
#
# Decoder handles four feature distribution types detected from training data:
#   'bernoulli'  → near-binary features (≥90% zeros, ≤3 unique values)
#   'zi_lognorm' → zero-inflated continuous (≥30% zeros)
#   'lognormal'  → right-skewed continuous (skew > 1.0, non-negative)
#   'continuous' → everything else (rank features, Gaussian-like)
#
# The encoder is the shared REncoder from rvae.py.
# Imported here so client_EFL only needs to import from models.

from rvae import REncoder as _REncoder   # GRU encoder shared with RVAE


def detect_zi_feature_types(X_np: np.ndarray,
                             bernoulli_zero_thresh: float = 0.90,
                             zi_zero_thresh:        float = 0.30,
                             lognormal_skew_thresh: float = 1.0,
                             rank_col_indices:      list  = None
                             ) -> dict:
    """
    Inspect a (N, T, F) array (pre-normalisation, raw values expected) and
    assign each of the F features to one distribution type.

    Parameters
    ----------
    X_np               : (N, T, F) float32 — raw (un-normalised) training windows
    bernoulli_zero_thresh : fraction of zeros above which a near-binary feature
                            is treated as Bernoulli (default 0.90)
    zi_zero_thresh     : fraction of zeros above which a feature gets a ZI gate
                         (default 0.30)
    lognormal_skew_thresh : skewness threshold for lognormal treatment (default 1.0)
    rank_col_indices   : list[int] — column indices to force to 'continuous'
                         (e.g. rank features that are differenced externally)

    Returns
    -------
    dict with keys 'bernoulli', 'zi_lognorm', 'lognormal', 'continuous',
    each mapping to a list of feature column indices.
    """
    from scipy.stats import skew as _skew

    rank_set = set(rank_col_indices or [])
    N, T, F  = X_np.shape
    flat     = X_np.reshape(-1, F)   # (N*T, F)

    types: dict[str, list[int]] = {
        'bernoulli' : [],
        'zi_lognorm': [],
        'lognormal' : [],
        'continuous': [],
    }

    for f in range(F):
        if f in rank_set:
            types['continuous'].append(f)
            continue
        col       = flat[:, f]
        n_unique  = len(np.unique(col))
        zero_pct  = float((col == 0).mean())
        col_skew  = float(_skew(col))

        if zero_pct >= bernoulli_zero_thresh and n_unique <= 3:
            types['bernoulli'].append(f)
        elif zero_pct >= zi_zero_thresh:
            types['zi_lognorm'].append(f)
        elif col.min() >= 0 and col_skew > lognormal_skew_thresh:
            types['lognormal'].append(f)
        else:
            types['continuous'].append(f)

    return types


class ZIRVAEDecoder(nn.Module):
    """
    GRU decoder with per-feature-type output heads.

    Produces:
        raw_out    : (B, T, F)  — sigmoid applied to all non-Bernoulli features;
                                  raw logits kept for Bernoulli (BCE-with-logits loss)
        gate_logit : (B, T, S) or None — ZI gate logits for zi_lognorm features
    """

    def __init__(self, latent_dim: int, hidden_dim: int, n_features: int,
                 window_size: int, feat_type_idx: dict, n_layers: int = 1):
        super().__init__()
        self.latent_dim    = latent_dim   # stored so generate() can sample z
        self.window_size   = window_size
        self.n_layers      = n_layers
        self.hidden_dim    = hidden_dim
        self.feat_type_idx = feat_type_idx
        self.n_features    = n_features

        self.fc_h0    = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_input = nn.Linear(latent_dim, n_features)
        self.gru      = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_features)

        zi_idx = feat_type_idx.get('zi_lognorm', [])
        self.gate_fc = nn.Linear(hidden_dim, len(zi_idx)) if zi_idx else None

    def forward(self, z: torch.Tensor):
        B          = z.size(0)
        h0         = self.fc_h0(z).view(self.n_layers, B, self.hidden_dim)
        # repeat z as input at every timestep so GRU can condition each step
        inp        = self.fc_input(z).unsqueeze(1).repeat(1, self.window_size, 1)
        gru_out, _ = self.gru(inp, h0)
        raw_out    = self.fc_out(gru_out)       # (B, T, F) — raw

        out = torch.sigmoid(raw_out)            # default: sigmoid for [0,1]

        # Bernoulli features stay as raw logits (loss uses BCE-with-logits)
        for f in self.feat_type_idx.get('bernoulli', []):
            out[:, :, f] = raw_out[:, :, f]

        gate_logit = self.gate_fc(gru_out) if self.gate_fc is not None else None
        return out, gate_logit

    @torch.no_grad()
    def generate(self, n: int, device: str = 'cpu') -> np.ndarray:
        """
        Sample n windows directly from the decoder (no encoder needed).
        z ~ N(0, I) is sampled here, so the server can call this without
        having access to the encoder or any real client data.
        Returns (n, T, F) float32 in [0, 1].
        """
        self.eval()
        z              = torch.randn(n, self.latent_dim, device=device)
        out, gate_logit = self.forward(z)
        result = out.clone()

        for f in self.feat_type_idx.get('bernoulli', []):
            result[:, :, f] = torch.bernoulli(torch.sigmoid(out[:, :, f]))

        zi_idx = self.feat_type_idx.get('zi_lognorm', [])
        if gate_logit is not None and zi_idx:
            gate_prob = torch.sigmoid(gate_logit)
            gate      = torch.bernoulli(gate_prob).bool()
            for s_local, s_global in enumerate(zi_idx):
                result[:, :, s_global] = torch.where(
                    gate[:, :, s_local],
                    out[:, :, s_global],
                    torch.zeros_like(out[:, :, s_global]),
                )

        return result.cpu().numpy().astype(np.float32)


class ZIRVAE(nn.Module):
    """
    Zero-Inflated Recurrent VAE.

    Encoder  : shared GRU REncoder from rvae.py
    Decoder  : ZIRVAEDecoder with per-feature-type heads
    Sampling : Bernoulli gate applied at synthesis time to zi_lognorm features
    """

    def __init__(self, n_features: int, window_size: int,
                 hidden_dim: int, latent_dim: int,
                 feat_type_idx: dict, n_layers: int = 1):
        super().__init__()
        self.latent_dim    = latent_dim
        self.feat_type_idx = feat_type_idx
        self.encoder  = _REncoder(n_features, hidden_dim, latent_dim, n_layers)
        self.decoder  = ZIRVAEDecoder(latent_dim, hidden_dim, n_features,
                                      window_size, feat_type_idx, n_layers)

    def forward(self, x: torch.Tensor):
        mu, log_var     = self.encoder(x)
        z               = self.encoder.reparameterize(mu, log_var)
        out, gate_logit = self.decoder(z)
        return out, gate_logit, mu, log_var

    @torch.no_grad()
    def generate(self, n: int, device: str = 'cpu') -> np.ndarray:
        """
        Sample n windows from the prior.  Returns (n, T, F) float32 in [0, 1].

        Bernoulli features  → Bernoulli sample from sigmoid(logit)
        ZI-lognorm features → gate * value  (zero if gate=0)
        All other features  → sigmoid output directly
        """
        self.eval()
        z              = torch.randn(n, self.latent_dim, device=device)
        out, gate_logit = self.decoder(z)
        result = out.clone()

        for f in self.feat_type_idx.get('bernoulli', []):
            result[:, :, f] = torch.bernoulli(torch.sigmoid(out[:, :, f]))

        zi_idx = self.feat_type_idx.get('zi_lognorm', [])
        if gate_logit is not None and zi_idx:
            gate_prob = torch.sigmoid(gate_logit)
            gate      = torch.bernoulli(gate_prob).bool()
            for s_local, s_global in enumerate(zi_idx):
                result[:, :, s_global] = torch.where(
                    gate[:, :, s_local],
                    out[:, :, s_global],
                    torch.zeros_like(out[:, :, s_global]),
                )

        return result.cpu().numpy().astype(np.float32)


def _zirvae_loss(x: torch.Tensor, decoder_out: torch.Tensor,
                 gate_logit, mu: torch.Tensor, log_var: torch.Tensor,
                 feat_type_idx: dict, kl_weight: float,
                 loss_factor: float, free_bits: float):
    recon   = torch.tensor(0.0, device=x.device)
    n_terms = 0

    for f in feat_type_idx.get('bernoulli', []):
        target = (x[:, :, f] > 0).float()
        recon  = recon + F.binary_cross_entropy_with_logits(
            decoder_out[:, :, f], target, reduction='mean')
        n_terms += 1

    zi_idx = feat_type_idx.get('zi_lognorm', [])
    if gate_logit is not None and zi_idx:
        for s_local, s_global in enumerate(zi_idx):
            gate_label = (x[:, :, s_global] > 0).float()
            recon = recon + 0.5 * F.binary_cross_entropy_with_logits(
                gate_logit[:, :, s_local], gate_label, reduction='mean')
            mask = gate_label.bool()
            if mask.any():
                recon = recon + 0.5 * F.mse_loss(
                    decoder_out[:, :, s_global][mask],
                    x[:, :, s_global][mask], reduction='mean')
            n_terms += 1

    for type_key in ('lognormal', 'continuous'):
        for f in feat_type_idx.get(type_key, []):
            recon   = recon + F.mse_loss(
                decoder_out[:, :, f], x[:, :, f], reduction='mean')
            n_terms += 1

    # loss_factor=1 keeps recon and KL on the same scale so KL signal is not drowned out
    recon_total    = recon / max(n_terms, 1)
    kl_elem        = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, D)
    # clamp per-sample per-dim before averaging — stronger floor than averaging first
    kl_loss        = torch.clamp(kl_elem, min=free_bits).sum(dim=-1).mean()
    return recon_total + kl_weight * kl_loss, \
           recon_total.item(), kl_loss.item()


def _cyclical_kl(epoch, n_epochs, n_cycles=4):
    # Ramp 0→1 over first half of each cycle, stay at 1 for second half.
    # With n_cycles=1 this is a single slow warmup — encoder can't collapse
    # because KL is always increasing, never reset to 0 mid-training.
    cycle_len = max(n_epochs / n_cycles, 1)
    pos       = (epoch % cycle_len) / cycle_len
    return min(1.0, pos * 2.0)


def _monotone_kl(epoch, n_epochs, warmup_frac=0.3):
    """
    Linear warmup from 0→1 over the first `warmup_frac` of training,
    then stays at 1.  No resets — prevents cyclical collapse.
    """
    warmup_epochs = max(1, int(n_epochs * warmup_frac))
    return min(1.0, epoch / warmup_epochs)


def train_zirvae(X_np: np.ndarray,
                 window_size: int,
                 n_features: int,
                 feat_type_idx: dict,
                 hidden_dim: int  = 128,
                 latent_dim: int  = 64,
                 n_layers: int    = 1,
                 epochs: int      = 300,
                 batch_size: int  = 256,
                 lr: float        = 1e-3,
                 noise_std: float = 0.10,
                 free_bits: float = 1.0,
                 n_cycles: int    = 4,
                 device: str      = 'cpu',
                 client_id: int   = 0) -> 'ZIRVAE':
    """
    Train a ZIRVAE on (N, T, F) float32 windows.

    Parameters
    ----------
    X_np          : (N, T, F) or (N, T*F) — automatically reshaped
    feat_type_idx : output of detect_zi_feature_types()

    Returns a trained ZIRVAE in eval mode.
    """
    X = X_np.reshape(-1, window_size, n_features).astype(np.float32)
    X_t = torch.tensor(X, device=device)

    model = ZIRVAE(n_features, window_size, hidden_dim,
                   latent_dim, feat_type_idx, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True,
    )
    bern_idx    = feat_type_idx.get('bernoulli', [])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    recon_history: list[float] = []

    model.train()
    bar = tqdm(range(epochs), desc=f"Client {client_id} ZI-RVAE",
               position=client_id, leave=True, dynamic_ncols=True)
    for epoch in bar:
        kl_weight = _monotone_kl(epoch, epochs, warmup_frac=0.3)
        e_loss = e_recon = e_kl = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()
            if noise_std > 0:
                noise = noise_std * torch.randn_like(batch)
                if bern_idx:
                    noise[:, :, bern_idx] = 0.0
                noisy = (batch + noise).clamp(0, 1)
            else:
                noisy = batch

            decoder_out, gate_logit, mu, log_var = model(noisy)
            loss, r, k = _zirvae_loss(
                batch, decoder_out, gate_logit, mu, log_var,
                feat_type_idx, kl_weight * 0.01, 1.0, free_bits,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            e_loss += loss.item(); e_recon += r; e_kl += k

        scheduler.step()
        n_b = max(len(loader), 1)
        epoch_recon = e_recon / n_b
        recon_history.append(epoch_recon)
        bar.set_postfix(loss=f"{e_loss/n_b:.4f}",
                        recon=f"{epoch_recon:.4f}",
                        kl=f"{e_kl/n_b:.4f}",
                        kl_w=f"{kl_weight:.2f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    return model, recon_history


# ── EFL: Student Model (permanent 2-dim binary classifier) ───────────────────

class StudentModel(nn.Module):
    """
    Lightweight binary LSTM classifier for edge clients.

    Output is always 2-dimensional (benign vs. attack) regardless of
    how many attack classes the Teacher has seen.  The collapsed Teacher
    distribution (Definition 1 in the EFL spec) is always projected to
    2 dims before comparison, so this head never needs to grow.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, fc_hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 2)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """x: (B, T, F)  →  logits: (B, 2)"""
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        feat = F.relu(self.fc1(out[:, -1, :]))
        return self.fc2(feat)   # (B, 2) raw logits


# ── EFL: Teacher Model (growing multiclass MLP on server) ────────────────────

class TeacherModel(nn.Module):
    """
    Multiclass LSTM Teacher residing on the server.

    Input  : (B, T, F) temporal windows from pool decoders
    Output : K logits, one per generator in the pool

    Uses a 2-layer LSTM encoder — the last hidden state captures
    temporal structure that a flat MLP would lose. The output head
    grows via grow_to_k() whenever a new decoder is admitted to the pool.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_classes: int = 1, n_features: int = None,
                 window_size: int = None):
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        # n_features and window_size used to reshape flat input if needed
        self.n_features  = n_features
        self.window_size = window_size

        self.lstm = nn.LSTM(
            input_size  = n_features if n_features else input_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.2,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, F)  or  (B, T*F) flat — auto-reshaped if needed.
        Returns logits (B, K).
        """
        if x.dim() == 2:
            # flat (B, T*F) → (B, T, F)
            B = x.size(0)
            x = x.reshape(B, self.window_size, self.n_features)
        _, (h_n, _) = self.lstm(x)   # h_n: (2, B, hidden_dim)
        out = self.norm(h_n[-1])      # last layer hidden state: (B, hidden_dim)
        return self.head(out)

    def grow_to_k(self, new_n_classes: int):
        """
        Expand output head from current size to new_n_classes.
        Existing neuron weights are preserved.
        """
        old_head = self.head
        old_k    = old_head.out_features
        if new_n_classes <= old_k:
            return
        new_head = nn.Linear(self.hidden_dim, new_n_classes)
        with torch.no_grad():
            new_head.weight[:old_k] = old_head.weight
            new_head.bias[:old_k]   = old_head.bias
        self.head = new_head


# ═══════════════════════════════════════════════════════════════════════════════
# TabDDPM — Tabular Denoising Diffusion Probabilistic Model
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture
# ─────────────
#   • Gaussian DDPM with T=1000 steps and a cosine noise schedule.
#   • Denoiser: residual MLP operating on flat (T*F,) vectors.
#     Sequences are flattened before diffusion and reshaped back after sampling.
#   • Server interface: TabDDPMGenerator.generate(n, device) → (N, T, F)
#     identical to ZIRVAEDecoder.generate() so the rest of the pipeline is
#     unchanged.
#
# Reference: Kotelnikov et al. "TabDDPM: Modelling Tabular Data with Diffusion
#            Models" (2023).  This is a clean re-implementation using only
#            PyTorch — no external library required.
# ═══════════════════════════════════════════════════════════════════════════════

class _SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding (Vaswani et al.)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


class _ResidualBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1    = nn.Linear(d_in, d_out)
        self.fc2    = nn.Linear(d_out, d_out)
        self.t_proj = nn.Linear(t_dim, d_out)
        self.norm1  = nn.LayerNorm(d_out)
        self.norm2  = nn.LayerNorm(d_out)
        self.drop   = nn.Dropout(dropout)
        self.skip   = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.fc1(x))) + self.t_proj(t_emb)
        h = self.drop(F.silu(self.norm2(self.fc2(h))))
        return h + self.skip(x)


class TabDDPMDenoiser(nn.Module):
    """
    Residual MLP denoiser: predicts noise ε given noisy input x_t and step t.
    Input/output dim = T * F (flat sequence).
    """
    def __init__(self, data_dim: int, hidden_dim: int = 512,
                 n_layers: int = 4, t_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.t_emb   = _SinusoidalEmbedding(t_dim)
        self.t_mlp   = nn.Sequential(nn.Linear(t_dim, t_dim * 2), nn.SiLU(),
                                     nn.Linear(t_dim * 2, t_dim))
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.blocks  = nn.ModuleList([
            _ResidualBlock(hidden_dim, hidden_dim, t_dim, dropout)
            for _ in range(n_layers)
        ])
        self.out     = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(self.t_emb(t))          # (B, t_dim)
        h     = self.input_proj(x)                  # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out(h)                          # (B, data_dim)


class TabDDPMGenerator:
    """
    Full DDPM wrapper: noise schedule, forward/reverse process, sampling.

    generate(n, device) → (N, T, F) float32  — same interface as ZIRVAEDecoder.
    save() / load() persist the denoiser weights for the server pool.
    """

    def __init__(self, data_dim: int, window_size: int, n_features: int,
                 n_steps: int = 1000, hidden_dim: int = 512,
                 n_layers: int = 4, t_dim: int = 128):
        self.data_dim    = data_dim       # T * F
        self.window_size = window_size
        self.n_features  = n_features
        self.n_steps     = n_steps

        self.denoiser = TabDDPMDenoiser(data_dim, hidden_dim, n_layers, t_dim)

        # cosine noise schedule (Nichol & Dhariwal 2021)
        steps  = np.arange(n_steps + 1)
        alphas = np.cos(((steps / n_steps) + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas  = np.clip(1 - alphas[1:] / alphas[:-1], 0, 0.999)

        alphas_bar = np.cumprod(1.0 - betas)
        self._betas          = torch.tensor(betas,      dtype=torch.float32)
        self._alphas_bar     = torch.tensor(alphas_bar, dtype=torch.float32)
        self._sqrt_ab        = torch.tensor(np.sqrt(alphas_bar),       dtype=torch.float32)
        self._sqrt_one_m_ab  = torch.tensor(np.sqrt(1 - alphas_bar),   dtype=torch.float32)

    def to(self, device):
        self.denoiser        = self.denoiser.to(device)
        self._betas          = self._betas.to(device)
        self._alphas_bar     = self._alphas_bar.to(device)
        self._sqrt_ab        = self._sqrt_ab.to(device)
        self._sqrt_one_m_ab  = self._sqrt_one_m_ab.to(device)
        self._device         = device
        return self

    def parameters(self):
        return self.denoiser.parameters()

    def train(self):
        self.denoiser.train()

    def eval(self):
        self.denoiser.eval()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε."""
        sa  = self._sqrt_ab[t][:, None]
        sma = self._sqrt_one_m_ab[t][:, None]
        return sa * x0 + sma * noise

    def p_losses(self, x0: torch.Tensor) -> torch.Tensor:
        """MSE between predicted noise and actual noise (simple loss)."""
        B      = x0.size(0)
        device = x0.device
        t      = torch.randint(0, self.n_steps, (B,), device=device)
        noise  = torch.randn_like(x0)
        x_t    = self.q_sample(x0, t, noise)
        pred   = self.denoiser(x_t, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def generate(self, n: int, device: str = 'cpu') -> np.ndarray:
        """
        DDPM reverse process: x_T ~ N(0,I) → x_0.
        Returns (N, T, F) float32 clipped to [0, 1].
        """
        self.denoiser.eval()
        dev = torch.device(device)
        x   = torch.randn(n, self.data_dim, device=dev)

        for step in reversed(range(self.n_steps)):
            t_batch = torch.full((n,), step, device=dev, dtype=torch.long)
            beta_t  = self._betas[step]
            ab_t    = self._alphas_bar[step]
            ab_prev = self._alphas_bar[step - 1] if step > 0 else torch.tensor(1.0, device=dev)

            eps_pred = self.denoiser(x, t_batch)

            # predict x0 from x_t and predicted noise
            x0_pred  = (x - self._sqrt_one_m_ab[step] * eps_pred) / self._sqrt_ab[step]
            x0_pred  = x0_pred.clamp(-1, 1)

            # posterior mean
            coef1 = (torch.sqrt(ab_prev) * beta_t) / (1 - ab_t)
            coef2 = (torch.sqrt(1 - beta_t) * (1 - ab_prev)) / (1 - ab_t)
            mean  = coef1 * x0_pred + coef2 * x

            if step > 0:
                var = beta_t * (1 - ab_prev) / (1 - ab_t)
                x   = mean + torch.sqrt(var) * torch.randn_like(mean)
            else:
                x   = mean

        # reshape flat → (N, T, F) and clip to [0, 1]
        out = x.reshape(n, self.window_size, self.n_features)
        out = out.clamp(0.0, 1.0)
        return out.cpu().numpy().astype(np.float32)

    def state_dict(self):
        return self.denoiser.state_dict()

    def load_state_dict(self, sd):
        self.denoiser.load_state_dict(sd)


def train_tabddpm(X_np: np.ndarray,
                  window_size: int,
                  n_features: int,
                  epochs: int       = 500,
                  batch_size: int   = 256,
                  lr: float         = 2e-4,
                  hidden_dim: int   = 512,
                  n_layers: int     = 4,
                  n_steps: int      = 1000,
                  device: str       = 'cpu',
                  client_id: int    = 0) -> tuple:
    """
    Train a TabDDPM on (N, T, F) windows.
    Returns (TabDDPMGenerator, loss_history).
    """
    X = X_np.reshape(-1, window_size, n_features).astype(np.float32)
    data_dim = window_size * n_features

    # flatten to (N, T*F) for the MLP denoiser
    X_flat = X.reshape(len(X), data_dim)
    X_t    = torch.tensor(X_flat, dtype=torch.float32, device=device)

    gen = TabDDPMGenerator(data_dim, window_size, n_features,
                           n_steps=n_steps, hidden_dim=hidden_dim,
                           n_layers=n_layers).to(device)

    optimizer = torch.optim.AdamW(gen.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True,
    )

    loss_history: list[float] = []
    gen.train()

    bar = tqdm(range(epochs), desc=f"Client {client_id} TabDDPM",
               position=client_id, leave=True, dynamic_ncols=True)

    for epoch in bar:
        ep_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = gen.p_losses(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()
        avg = ep_loss / max(len(loader), 1)
        loss_history.append(avg)
        bar.set_postfix(loss=f"{avg:.4f}")

    gen.eval()
    return gen, loss_history