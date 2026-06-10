# lstmvae_toy.py
#
# Side-by-side comparison of TimeVAE (Conv1D) vs RVAE (GRU) vs ZI-RVAE on one CSV.
# Both use class-conditional training (one generator per class).
#
# ZI-RVAE = RVAE GRU encoder  +  Zero-Inflated decoder
#   Dense features  → Gaussian head (sigmoid)
#   Sparse features → Bernoulli gate head  +  Gaussian value head
#   During synthesis: output = gate * value  (exact zeros preserved)
#
# Outputs per model:
#   - KS test per feature
#   - TSTR / TRTS / Baseline F1
#   - Distribution overlay plots
#   - Correlation heatmaps
#   - Final side-by-side summary table
#
# Run:  python toy/lstmvae_toy.py

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F_torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)

from time_vae import train_time_vae, synthesize_time_vae
from rvae     import train_rvae, RVAE, REncoder

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'attack_data',
    'blackhole_var10_base',
    '10_features_timeseries_60_sec.csv'
)
WINDOW_SIZE = 10
N_SYNTH     = 1000
EPOCHS      = 300
DEVICE      = 'cpu'
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'lstmvae_toy_results')
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════

SPARSE_FEATS_THRESH = 0.30   # fraction of zeros → sparse feature
RANK_COLS           = ['rank', 'rank.1']   # monotonic counters → differenced


def load_and_window(csv_path, window_size):
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    feat_cols = [c for c in df.columns if c != 'label']

    # ── Differencing for rank features ───────────────
    # rank / rank.1 are near-monotonic global counters.
    # Their absolute value encodes time position, not behaviour.
    # First-differencing removes the trend and leaves the change rate,
    # which is stationary and modelable by the VAE.
    # The first row becomes NaN after diff → filled with 0.
    rank_cols_present = [c for c in RANK_COLS if c in feat_cols]
    if rank_cols_present:
        diff = df[rank_cols_present].diff().fillna(0)
        # sign-preserving log compression — compresses heavy tails, keeps zeros
        df[rank_cols_present] = np.sign(diff) * np.log1p(np.abs(diff))
        print(f"   Differenced + sign-log1p compressed rank features: {rank_cols_present}")

    # Identify sparse features on the full data (before split)
    sparse_cols = [
        c for c in feat_cols
        if pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])
        if (df[c] == 0).mean() > SPARSE_FEATS_THRESH
    ]
    sparse_idx = [feat_cols.index(c) for c in sparse_cols]
    print(f"   log1p applied to {len(sparse_cols)} sparse features: {sparse_cols}")

    # Apply log1p in-place (values are non-negative RPL metrics)
    if sparse_cols:
        df[sparse_cols] = np.log1p(df[sparse_cols])

    split    = int(len(df) * 0.8)
    tr, te   = df.iloc[:split].copy(), df.iloc[split:].copy()
    g_min    = tr[feat_cols].min()
    g_max    = tr[feat_cols].max()
    denom    = (g_max - g_min).replace(0, 1)

    for d in [tr, te]:
        d[feat_cols] = ((d[feat_cols] - g_min) / denom).clip(0, 1).fillna(0)

    # ── Symmetric rescaling for rank columns ─────────
    # After sign-log1p, rank cols span [-A, +A] with a spike at 0.
    # Standard min-max maps 0 to an interior value (~0.56), creating
    # a hidden spike that confuses the VAE.
    # Fix: scale by abs-max so 0 → 0.5, negatives → [0, 0.5), positives → (0.5, 1]
    for d in [tr, te]:
        for col in rank_cols_present:
            abs_max = tr[col].abs().max()
            if abs_max > 0:
                d[col] = (d[col] / (2 * abs_max) + 0.5).clip(0, 1)

    def _windows(d):
        X, y = [], []
        vals   = d[feat_cols].values.astype(np.float32)
        labels = d['label'].values.astype(int)
        for i in range(len(vals) - window_size):
            X.append(vals[i:i+window_size])
            y.append(labels[i + window_size - 1])
        return np.array(X, np.float32), np.array(y, np.int64)

    X_tr, y_tr = _windows(tr)
    X_te, y_te = _windows(te)
    return X_tr, y_tr, X_te, y_te, feat_cols, sparse_idx


def ks_table(real_flat, syn_flat, col_names):
    rows = []
    for i, col in enumerate(col_names):
        s, p = stats.ks_2samp(real_flat[:, i], syn_flat[:, i])
        rows.append({'feature': col,
                     'ks_stat': round(float(s), 4),
                     'p_value': round(float(p), 4),
                     'similar': bool(p > 0.05)})
    return pd.DataFrame(rows).sort_values('ks_stat', ascending=False)


def clf_score(X_tr, y_tr, X_te, y_te):
    clf = LogisticRegression(max_iter=1000, C=1.0,
                              solver='lbfgs', random_state=42)
    try:
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        return float(accuracy_score(y_te, preds)), \
               float(f1_score(y_te, preds, average='macro', zero_division=0))
    except Exception:
        return 0.0, 0.0


# ═══════════════════════════════════════════════════
# ZI-RVAE — Zero-Inflated decoder on top of GRU encoder
# ═══════════════════════════════════════════════════
#
# Encoder : reuses RVAE's GRU encoder (REncoder from rvae.py)
#           (B, T, F) → GRU → last hidden → (mu, log_var)
#
# Decoder : Zero-Inflated two-head design
#   z → fc_h0 (GRU init state) + fc_input (repeated T times)
#     → GRU → fc_out → sigmoid        : Gaussian head  (B, T, F)
#     → gate_fc → sigmoid             : Bernoulli gate  (B, T, S)  sparse only
#
# Synthesis:
#   gate ~ Bernoulli(gate_prob)
#   output[:, :, sparse_idx[s]] = gate * gaussian_value
#                                = 0      if gate == 0
#                                = value  if gate == 1


class ZIRVAEDecoder(nn.Module):
    """
    GRU decoder with Zero-Inflated output heads.

    For all features     : Gaussian head  (sigmoid)  → value in [0, 1]
    For sparse features  : Bernoulli gate (sigmoid)  → P(feature > 0)

    Synthesis: output = gate * gaussian_value  on sparse dims.
    """

    def __init__(self, latent_dim: int, hidden_dim: int,
                 n_features: int, window_size: int,
                 sparse_idx: list, n_layers: int = 1):
        super().__init__()
        self.window_size = window_size
        self.n_layers    = n_layers
        self.hidden_dim  = hidden_dim
        self.sparse_idx  = sparse_idx

        # Standard GRU decoder path (same as RVAE RDecoder)
        self.fc_h0    = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_input = nn.Linear(latent_dim, n_features)
        self.gru      = nn.GRU(n_features, hidden_dim, n_layers,
                               batch_first=True)
        self.fc_out   = nn.Linear(hidden_dim, n_features)   # Gaussian head

        # Bernoulli gate head — one logit per sparse feature per timestep
        if sparse_idx:
            self.gate_fc = nn.Linear(hidden_dim, len(sparse_idx))
        else:
            self.gate_fc = None

    def forward(self, z: torch.Tensor):
        B   = z.size(0)
        h0  = self.fc_h0(z).view(self.n_layers, B, self.hidden_dim)
        inp = self.fc_input(z).unsqueeze(1).expand(-1, self.window_size, -1)

        gru_out, _ = self.gru(inp, h0)                       # (B, T, hidden)
        gauss_out  = torch.sigmoid(self.fc_out(gru_out))     # (B, T, F)

        if self.gate_fc is not None:
            gate_logit = self.gate_fc(gru_out)               # (B, T, S)
        else:
            gate_logit = None

        return gauss_out, gate_logit


class ZIRVAE(nn.Module):
    """
    Zero-Inflated Recurrent VAE.

    Encoder  : GRU (from rvae.py — REncoder)
    Decoder  : ZIRVAEDecoder (GRU + Gaussian head + Bernoulli gate head)
    """

    def __init__(self, n_features: int, window_size: int,
                 hidden_dim: int, latent_dim: int,
                 sparse_idx: list, n_layers: int = 1):
        super().__init__()
        self.n_features  = n_features
        self.window_size = window_size
        self.latent_dim  = latent_dim
        self.sparse_idx  = sparse_idx

        self.encoder = REncoder(n_features, hidden_dim, latent_dim, n_layers)
        self.decoder = ZIRVAEDecoder(latent_dim, hidden_dim, n_features,
                                     window_size, sparse_idx, n_layers)

    def forward(self, x: torch.Tensor):
        mu, log_var           = self.encoder(x)
        z                     = self.encoder.reparameterize(mu, log_var)
        gauss_out, gate_logit = self.decoder(z)
        return gauss_out, gate_logit, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu') -> np.ndarray:
        """
        Generate n windows.  Returns (n, T, F) float32 in [0, 1].

        For each sparse feature at each timestep:
            gate ~ Bernoulli(sigmoid(gate_logit))
            output = gate * gauss_value
        """
        z = torch.randn(n, self.latent_dim, device=device)
        gauss_out, gate_logit = self.decoder(z)

        out = gauss_out.clone()
        if gate_logit is not None:
            gate_prob = torch.sigmoid(gate_logit)              # (n, T, S)
            gate      = torch.bernoulli(gate_prob).bool()      # (n, T, S)
            for s_local, s_global in enumerate(self.sparse_idx):
                out[:, :, s_global] = torch.where(
                    gate[:, :, s_local],
                    gauss_out[:, :, s_global],
                    torch.zeros_like(gauss_out[:, :, s_global])
                )

        return out.cpu().numpy().astype(np.float32)


def zirvae_loss(x           : torch.Tensor,
                gauss_out   : torch.Tensor,
                gate_logit  : torch.Tensor,
                mu          : torch.Tensor,
                log_var     : torch.Tensor,
                sparse_idx  : list,
                kl_weight   : float = 1.0,
                loss_factor : float = 1.0,
                free_bits   : float = 0.5) -> tuple:
    """
    ELBO loss for ZI-RVAE.

    recon_mse : MSE over all (B, T, F)             — Gaussian reconstruction
    recon_bce : BCE on gate logits for sparse dims — Bernoulli gate
      gate_label = (x > 0).float()
    kl_loss   : free-bits KL (same as rvae_loss)
    """
    # ── Gaussian reconstruction ──────────────────────
    recon_mse = F_torch.mse_loss(gauss_out, x, reduction='mean')

    # ── Bernoulli gate loss on sparse features ───────
    recon_bce = torch.tensor(0.0, device=x.device)
    if gate_logit is not None and len(sparse_idx) > 0:
        for s_local, s_global in enumerate(sparse_idx):
            gate_label = (x[:, :, s_global] > 0).float()     # (B, T)
            gl         = gate_logit[:, :, s_local]            # (B, T)
            recon_bce  = recon_bce + F_torch.binary_cross_entropy_with_logits(
                gl, gate_label, reduction='mean'
            )

    recon_total = loss_factor * recon_mse + 0.1 * recon_bce

    # ── Free-bits KL ─────────────────────────────────
    kl_elementwise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_per_dim     = kl_elementwise.mean(dim=0)
    kl_loss        = torch.clamp(kl_per_dim, min=free_bits).sum()

    total = recon_total + kl_weight * kl_loss
    return total, recon_mse.item(), kl_loss.item()


def _cyclical_kl_weight(epoch, n_epochs, n_cycles=4,
                         min_w=0.0, max_w=1.0):
    cycle_len = max(n_epochs / n_cycles, 1)
    pos       = (epoch % cycle_len) / cycle_len
    ramp      = min(1.0, pos * 2.0)
    return min_w + (max_w - min_w) * ramp


def train_zirvae(X_np        : np.ndarray,
                 window_size : int,
                 n_features  : int,
                 sparse_idx  : list,
                 hidden_dim  : int   = 128,
                 latent_dim  : int   = 64,
                 n_layers    : int   = 1,
                 epochs      : int   = 300,
                 batch_size  : int   = 256,
                 lr          : float = 1e-3,
                 noise_std   : float = 0.10,
                 free_bits   : float = 1.0,
                 n_cycles    : int   = 4,
                 device      : str   = 'cpu') -> ZIRVAE:
    """
    Train a ZI-RVAE.

    X_np shape : (N, window_size * n_features)  — flat windows, as in train_rvae.
    sparse_idx : list of feature column indices that are zero-inflated.
    """
    X   = X_np.reshape(-1, window_size, n_features).astype(np.float32)
    X_t = torch.tensor(X, device=device)

    model     = ZIRVAE(n_features, window_size, hidden_dim, latent_dim,
                       sparse_idx, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=1e-5)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=True,
    )

    loss_factor = float(window_size * n_features)

    model.train()
    for epoch in range(epochs):
        kl_weight   = _cyclical_kl_weight(epoch, epochs, n_cycles=n_cycles)
        epoch_loss  = epoch_recon = epoch_kl = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()

            # Denoising: add noise to encoder input, decode clean target
            if noise_std > 0:
                noisy = batch + noise_std * torch.randn_like(batch)
                noisy = torch.clamp(noisy, 0.0, 1.0)
            else:
                noisy = batch

            gauss_out, gate_logit, mu, log_var = model(noisy)
            loss, recon_val, kl_val = zirvae_loss(
                x           = batch,
                gauss_out   = gauss_out,
                gate_logit  = gate_logit,
                mu          = mu,
                log_var     = log_var,
                sparse_idx  = sparse_idx,
                kl_weight   = kl_weight,
                loss_factor = loss_factor,
                free_bits   = free_bits,
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
                f"  Epoch [{epoch+1:>4}/{epochs}] "
                f"loss={epoch_loss/n_b:.4f}  "
                f"recon={epoch_recon/n_b:.4f}  "
                f"kl={epoch_kl/n_b:.4f}  "
                f"kl_w={kl_weight:.2f}  "
                f"free_bits={free_bits:.1f}"
            )

    model.eval()
    return model


# ═══════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════
print("\n── 1. Loading data ──────────────────────────────")
X_tr, y_tr, X_te, y_te, feat_cols, sparse_idx = load_and_window(
    CSV_PATH, WINDOW_SIZE
)
N, T, F = X_tr.shape
print(f"   Train: {X_tr.shape}  Test: {X_te.shape}")
print(f"   Label dist train: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
print(f"   Sparse idx: {sparse_idx}  ({[feat_cols[i] for i in sparse_idx]})")

# flat versions for evaluation (mean over T axis)
X_tr_flat = X_tr.mean(axis=1)   # (N_tr, F)
X_te_flat = X_te.mean(axis=1)   # (N_te, F)

# ── shared config ─────────────────────────────────
latent_dim = max(4, min((T * F) // 10, 32))
hidden_dim = max(64, T * F * 4)
free_bits  = round(min(0.5, 8.0 / latent_dim), 4)
batch_size = min(256, max(32, N // 20))
print(f"   Shared: latent={latent_dim}  hidden={hidden_dim}  "
      f"free_bits={free_bits}  batch={batch_size}")


# ═══════════════════════════════════════════════════
# 2. TRAIN — all three models, one per class
# ═══════════════════════════════════════════════════
results = {}   # model_name → { ks_df, tstr, trts, base, X_syn, y_syn }

for model_name, train_fn in [
    ('TimeVAE',   'timevae'),
    ('RVAE (GRU)', 'rvae'),
    ('ZI-RVAE',   'zirvae'),
]:
    print(f"\n{'='*55}")
    print(f"  Training {model_name}")
    print(f"{'='*55}")

    syn_X_parts, syn_y_parts = [], []

    for cls in [0, 1]:
        X_cls = X_tr[y_tr == cls]
        print(f"\n  Class {cls}: {len(X_cls)} windows")
        if len(X_cls) < batch_size:
            print(f"  [skip] too few samples")
            continue

        if train_fn == 'timevae':
            model = train_time_vae(
                X_np         = X_cls,
                T=T, F=F,
                latent_dim   = latent_dim,
                hidden_dim   = hidden_dim,
                enc_filters  = (32, 64),
                kernel_size  = 3,
                epochs       = EPOCHS,
                batch_size   = batch_size,
                lr           = 1e-3,
                recon_weight = 2.0,
                kl_warmup    = 100,
                free_bits    = free_bits,
                device       = DEVICE,
            )
            X_syn = synthesize_time_vae(model, N_SYNTH, device=DEVICE)
            # X_syn: (N_SYNTH, T, F)

        elif train_fn == 'rvae':
            X_flat = X_cls.reshape(len(X_cls), -1)
            model  = train_rvae(
                X_np            = X_flat,
                window_size     = T,
                n_raw_features  = F,
                hidden_dim      = hidden_dim,
                latent_dim      = latent_dim,
                n_layers        = 1,
                epochs          = EPOCHS,
                batch_size      = batch_size,
                lr              = 1e-3,
                noise_std       = 0.05,
                free_bits       = free_bits,
                n_cycles        = 4,
                device          = DEVICE,
            )
            with torch.no_grad():
                z     = torch.randn(N_SYNTH, model.latent_dim)
                X_syn = model.decoder(z).numpy()   # (N_SYNTH, T, F)

        else:  # zirvae
            X_flat = X_cls.reshape(len(X_cls), -1)
            model  = train_zirvae(
                X_np        = X_flat,
                window_size = T,
                n_features  = F,
                sparse_idx  = sparse_idx,
                hidden_dim  = hidden_dim,
                latent_dim  = latent_dim,
                n_layers    = 1,
                epochs      = EPOCHS,
                batch_size  = batch_size,
                lr          = 1e-3,
                noise_std   = 0.05,
                free_bits   = free_bits,
                n_cycles    = 4,
                device      = DEVICE,
            )
            X_syn = model.sample(N_SYNTH, device=DEVICE)   # (N_SYNTH, T, F)

        syn_X_parts.append(X_syn)
        syn_y_parts.append(np.full(N_SYNTH, cls, dtype=np.int64))

    X_syn_all = np.concatenate(syn_X_parts, axis=0)   # (2*N_SYNTH, T, F)
    y_syn_all = np.concatenate(syn_y_parts, axis=0)

    # flat for evaluation
    X_syn_flat = X_syn_all.mean(axis=1)               # (2*N_SYNTH, F)

    # ── KS overall (mixed) ────────────────────────────
    ks_df = ks_table(X_tr_flat, X_syn_flat, feat_cols)

    # ── KS per class ──────────────────────────────────
    ks_per_class = {}
    for cls in [0, 1]:
        real_cls = X_tr_flat[y_tr == cls]
        syn_cls  = X_syn_flat[y_syn_all == cls]
        ks_per_class[cls] = ks_table(real_cls, syn_cls, feat_cols)

    _, tstr_f1 = clf_score(X_syn_flat, y_syn_all, X_te_flat, y_te)
    _, trts_f1 = clf_score(X_tr_flat,  y_tr,      X_syn_flat, y_syn_all)
    _, base_f1 = clf_score(X_tr_flat,  y_tr,      X_te_flat,  y_te)

    results[model_name] = {
        'ks_df'       : ks_df,
        'ks_per_class': ks_per_class,
        'tstr_f1'     : tstr_f1,
        'trts_f1'     : trts_f1,
        'base_f1'     : base_f1,
        'X_syn_flat'  : X_syn_flat,
        'y_syn'       : y_syn_all,
    }

    print(f"\n  {model_name} KS (overall) mean={ks_df['ks_stat'].mean():.4f}  "
          f"pass={ks_df['similar'].sum()}/{len(ks_df)}")
    for cls in [0, 1]:
        kc = ks_per_class[cls]
        print(f"  {model_name} KS class={cls}  mean={kc['ks_stat'].mean():.4f}  "
              f"pass={kc['similar'].sum()}/{len(kc)}  "
              f"passed={kc[kc['similar']]['feature'].tolist()}")
    print(f"  {model_name} TSTR f1={tstr_f1:.4f}  "
          f"TRTS f1={trts_f1:.4f}  Baseline f1={base_f1:.4f}")


# ═══════════════════════════════════════════════════
# 3. PLOTS
# ═══════════════════════════════════════════════════
print("\n── 3. Saving plots ──────────────────────────────")
model_names = list(results.keys())
colors      = {
    'TimeVAE'   : 'darkorange',
    'RVAE (GRU)': 'seagreen',
    'ZI-RVAE'   : 'mediumpurple',
}

real_df = pd.DataFrame(X_tr_flat, columns=feat_cols)

# ── 3a. Distribution overlays — one row per model ──
n_feat = len(feat_cols)
fig, axes = plt.subplots(
    len(model_names), n_feat,
    figsize=(2.8 * n_feat, 3.5 * len(model_names)),
    constrained_layout=True
)

for row_idx, mname in enumerate(model_names):
    X_syn_flat = results[mname]['X_syn_flat']
    syn_df     = pd.DataFrame(X_syn_flat, columns=feat_cols)
    ks_lu      = results[mname]['ks_df'].set_index('feature')

    for col_idx, col in enumerate(feat_cols):
        ax = axes[row_idx, col_idx]
        r  = real_df[col].values
        s  = syn_df[col].values

        ax.hist(r, bins=25, alpha=0.35, color='steelblue',
                density=True, label='Real')
        ax.hist(s, bins=25, alpha=0.35, color=colors[mname],
                density=True, label=mname)
        try:
            xs = np.linspace(min(r.min(), s.min()),
                             max(r.max(), s.max()), 150)
            ax.plot(xs, stats.gaussian_kde(r)(xs),
                    color='steelblue', lw=1.2)
            ax.plot(xs, stats.gaussian_kde(s)(xs),
                    color=colors[mname], lw=1.2)
        except Exception:
            pass

        ks_v = ks_lu.loc[col, 'ks_stat'] if col in ks_lu.index else float('nan')
        is_sparse = col in [feat_cols[i] for i in sparse_idx]
        ax.set_title(f"{'*' if is_sparse else ''}{col}\nKS={ks_v:.3f}",
                     fontsize=6)
        ax.tick_params(labelsize=5)
        if col_idx == 0:
            ax.set_ylabel(mname, fontsize=7)
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=5)

fig.suptitle(
    'Distribution Overlay — Real vs Synthetic  (* = sparse feature)\n'
    'Row 1: TimeVAE   Row 2: RVAE (GRU)   Row 3: ZI-RVAE',
    fontsize=10
)
path = os.path.join(OUT_DIR, 'distributions_comparison.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")

# ── 3b. KS bar chart — one subplot per model ───────
fig, axes = plt.subplots(1, len(model_names),
                          figsize=(8 * len(model_names), 4),
                          constrained_layout=True)
for ax, mname in zip(axes, model_names):
    ks_df  = results[mname]['ks_df'].sort_values('ks_stat', ascending=False)
    bar_colors = ['tomato' if not r else 'steelblue'
                  for r in ks_df['similar'].tolist()]
    ax.bar(ks_df['feature'], ks_df['ks_stat'], color=bar_colors)
    ax.axhline(0.05, color='black', linestyle='--', lw=1.2)
    ax.set_title(f'{mname}  —  mean KS={ks_df["ks_stat"].mean():.3f}  '
                 f'pass={ks_df["similar"].sum()}/{len(ks_df)}')
    ax.set_ylabel('KS Statistic')
    ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

fig.suptitle('KS Statistic per Feature  '
             '(blue=pass p>0.05, red=fail)', fontsize=11)
path = os.path.join(OUT_DIR, 'ks_comparison.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")

# ── 3c. Correlation heatmaps ───────────────────────
for mname in model_names:
    X_syn_flat = results[mname]['X_syn_flat']
    syn_df     = pd.DataFrame(X_syn_flat, columns=feat_cols)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5),
                                      constrained_layout=True)
    rc  = real_df.corr()
    sc  = syn_df.corr()
    dc  = (rc - sc).abs()
    kw  = dict(cmap='coolwarm', vmin=-1, vmax=1, square=True,
               annot=True, fmt='.2f', annot_kws={'size': 6},
               xticklabels=True, yticklabels=True)
    sns.heatmap(rc, ax=a1, **kw);  a1.set_title('Real')
    sns.heatmap(sc, ax=a2, **kw);  a2.set_title(f'Synthetic ({mname})')
    sns.heatmap(dc, ax=a3, cmap='Reds', vmin=0, vmax=1, square=True,
                annot=True, fmt='.2f', annot_kws={'size': 6},
                xticklabels=True, yticklabels=True)
    a3.set_title('|Difference|')
    fig.suptitle(f'Correlation — {mname}', fontsize=11)
    safe = mname.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    path = os.path.join(OUT_DIR, f'correlations_{safe}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path}")

# ── 3d. TSTR / TRTS bar comparison ─────────────────
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
x      = np.arange(3)
width  = 0.25
labels = ['Baseline\n(real→real)', 'TRTS\n(real→syn)', 'TSTR\n(syn→real)']
n_models = len(model_names)

for i, mname in enumerate(model_names):
    r    = results[mname]
    vals = [r['base_f1'], r['trts_f1'], r['tstr_f1']]
    off  = (i - (n_models - 1) / 2) * width
    bars = ax.bar(x + off, vals, width, label=mname,
                  color=colors[mname], alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Macro F1')
ax.axhline(0.5, color='red', linestyle='--', lw=1, label='random')
ax.legend()
ax.set_title('TSTR / TRTS / Baseline — TimeVAE vs RVAE (GRU) vs ZI-RVAE')
path = os.path.join(OUT_DIR, 'tstr_trts_comparison.png')
plt.savefig(path, dpi=120, bbox_inches='tight')
plt.close()
print(f"   Saved: {path}")


# ═══════════════════════════════════════════════════
# 4. SUMMARY TABLE
# ═══════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════")
print("  FINAL COMPARISON SUMMARY")
print("══════════════════════════════════════════════════")
header = (f"{'Metric':<28} {'TimeVAE':>12} "
          f"{'RVAE (GRU)':>12} {'ZI-RVAE':>12}")
print(header)
print('-' * len(header))

metrics = [
    ('Mean KS stat',     lambda r: f"{r['ks_df']['ks_stat'].mean():.4f}"),
    ('KS pass rate',     lambda r: f"{r['ks_df']['similar'].sum()}/{len(r['ks_df']['ks_stat'])}"),
    ('Baseline f1',      lambda r: f"{r['base_f1']:.4f}"),
    ('TRTS f1',          lambda r: f"{r['trts_f1']:.4f}"),
    ('TSTR f1',          lambda r: f"{r['tstr_f1']:.4f}"),
    ('TSTR/TRTS gap',    lambda r: f"{abs(r['trts_f1']-r['tstr_f1']):.4f}"),
]

for label, fn in metrics:
    tv_val   = fn(results['TimeVAE'])
    rvae_val = fn(results['RVAE (GRU)'])
    zi_val   = fn(results['ZI-RVAE'])
    print(f"  {label:<26} {tv_val:>12} {rvae_val:>12} {zi_val:>12}")

print(f"\n  Results saved to: {OUT_DIR}")
print("══════════════════════════════════════════════════\n")

# ═══════════════════════════════════════════════════
# 5. KS DETAILS — passed / failed features per model
# ═══════════════════════════════════════════════════
print("\n── KS Feature Breakdown ─────────────────────────")
ks_rows = []

for mname in model_names:
    print(f"\n  {'='*50}")
    print(f"  {mname}")
    print(f"  {'='*50}")

    # overall
    ks_df  = results[mname]['ks_df']
    passed = ks_df[ks_df['similar']]['feature'].tolist()
    failed = ks_df[~ks_df['similar']]['feature'].tolist()
    print(f"  OVERALL  — PASSED ({len(passed)}): {passed}")
    print(f"             FAILED ({len(failed)}): {failed}")

    for _, row in ks_df.iterrows():
        ks_rows.append({
            'model'  : mname,
            'class'  : 'overall',
            'feature': row['feature'],
            'ks_stat': row['ks_stat'],
            'p_value': row['p_value'],
            'passed' : row['similar'],
        })

    # per class
    for cls in [0, 1]:
        label = 'normal' if cls == 0 else 'attack'
        kc     = results[mname]['ks_per_class'][cls]
        passed = kc[kc['similar']]['feature'].tolist()
        failed = kc[~kc['similar']]['feature'].tolist()
        print(f"  Class {cls} ({label}) — PASSED ({len(passed)}): {passed}")
        print(f"               FAILED ({len(failed)}): {failed}")

        for _, row in kc.iterrows():
            ks_rows.append({
                'model'  : mname,
                'class'  : label,
                'feature': row['feature'],
                'ks_stat': row['ks_stat'],
                'p_value': row['p_value'],
                'passed' : row['similar'],
            })

ks_all = pd.DataFrame(ks_rows)
ks_csv = os.path.join(OUT_DIR, 'ks_feature_results.csv')
ks_all.to_csv(ks_csv, index=False)
print(f"\n  KS details saved to: {ks_csv}")
