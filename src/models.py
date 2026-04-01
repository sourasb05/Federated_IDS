import torch.nn as nn
import torch.nn.functional as F
import torch

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