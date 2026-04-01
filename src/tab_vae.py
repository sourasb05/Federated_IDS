# tab_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ─────────────────────────────────────────────
# Gumbel-Softmax
# ─────────────────────────────────────────────
def gumbel_softmax(logits, tau=0.2, hard=True):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = F.softmax((logits + g) / tau, dim=-1)
    if hard:
        idx    = y.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
        y      = (y_hard - y).detach() + y
    return y


# ─────────────────────────────────────────────
# Transformer
# ─────────────────────────────────────────────
class TabTransformer:
    """
    Fits and transforms tabular data for Tab-VAE.
    Handles continuous (MSN) and categorical (OHE) columns.
    """
    def __init__(self, continuous_cols, categorical_cols,
                 n_gmm_components=5):
        self.continuous_cols  = continuous_cols
        self.categorical_cols = categorical_cols
        self.n_gmm            = n_gmm_components

        self.gmms      = {}   # col → fitted BayesianGMM
        self.ohe       = {}   # col → fitted OneHotEncoder
        self.scalers   = {}   # col → StandardScaler fallback
        self.col_info  = {}   # col → {'type', 'n_modes'/'n_cats', 'slice'}

    def fit(self, df: pd.DataFrame):
        ptr = 0
        for col in self.continuous_cols:
            data = df[col].values.astype(float).reshape(-1, 1)
            gmm  = BayesianGaussianMixture(
                n_components=self.n_gmm,
                weight_concentration_prior=0.001,
                max_iter=100, random_state=42
            )
            gmm.fit(data)
            active  = gmm.weights_ > 0.01
            n_modes = int(active.sum())
            n_modes = max(n_modes, 1)

            self.gmms[col] = (gmm, active)
            dim = 1 + n_modes   # alpha + beta one-hot
            self.col_info[col] = {
                'type': 'continuous',
                'n_modes': n_modes,
                'slice': (ptr, ptr + dim)
            }
            ptr += dim

        for col in self.categorical_cols:
            data = df[col].values.astype(str).reshape(-1, 1)
            enc  = OneHotEncoder(sparse_output=False,
                                 handle_unknown='ignore')
            enc.fit(data)
            n_cats = len(enc.categories_[0])
            self.ohe[col] = enc
            self.col_info[col] = {
                'type': 'categorical',
                'n_cats': n_cats,
                'slice': (ptr, ptr + n_cats)
            }
            ptr += n_cats

        self.total_dim = ptr

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        N   = len(df)
        out = np.zeros((N, self.total_dim), dtype=np.float32)

        for col in self.continuous_cols:
            gmm, active = self.gmms[col]
            info = self.col_info[col]
            s, e = info['slice']
            n_m  = info['n_modes']

            data  = df[col].values.astype(float)
            probs = gmm.predict_proba(data.reshape(-1,1))[:,active]
            if probs.shape[1] == 0:
                probs = np.ones((N, 1))
            mode_idx = probs.argmax(axis=1)

            means = gmm.means_[active].flatten()
            stds  = np.sqrt(
                gmm.covariances_[active]
            ).flatten()
            stds  = np.clip(stds, 1e-6, None)

            alpha = (data - means[mode_idx]) / (4 * stds[mode_idx])
            alpha = np.clip(alpha, -1, 1)
            beta  = np.eye(n_m)[mode_idx]

            out[:, s]       = alpha
            out[:, s+1:e]   = beta

        for col in self.categorical_cols:
            info = self.col_info[col]
            s, e = info['slice']
            data = df[col].values.astype(str).reshape(-1,1)
            out[:, s:e] = self.ohe[col].transform(data)

        return out

    def inverse_transform(self,
                          alphas, betas, gammas,
                          ) -> pd.DataFrame:
        rows = {}

        for i, col in enumerate(self.continuous_cols):
            gmm, active = self.gmms[col]
            info = self.col_info[col]
            n_m  = info['n_modes']

            a_vals   = alphas[i].cpu().numpy().flatten()
            b_idx    = betas[i].cpu().numpy().argmax(axis=1)
            b_idx    = np.clip(b_idx, 0, n_m - 1)

            means = gmm.means_[active].flatten()
            stds  = np.sqrt(
                gmm.covariances_[active]
            ).flatten()
            stds  = np.clip(stds, 1e-6, None)

            rows[col] = a_vals * 4 * stds[b_idx] + means[b_idx]

        for i, col in enumerate(self.categorical_cols):
            enc   = self.ohe[col]
            g_idx = gammas[i].cpu().numpy().argmax(axis=1)
            cats  = enc.categories_[0]
            g_idx = np.clip(g_idx, 0, len(cats) - 1)
            rows[col] = cats[g_idx]

        return pd.DataFrame(rows)

    def get_decoder_info(self):
        """Returns dims needed to build decoder heads."""
        cont_dims = []
        for col in self.continuous_cols:
            n_m = self.col_info[col]['n_modes']
            cont_dims.append((1, n_m))   # (alpha_dim, beta_dim)

        cat_dims = []
        for col in self.categorical_cols:
            cat_dims.append(self.col_info[col]['n_cats'])

        return cont_dims, cat_dims


# ─────────────────────────────────────────────
# Encoder / Decoder / Tab-VAE
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h       = self.net(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,
                 cont_dims, cat_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.alpha_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in cont_dims
        ])
        self.beta_heads  = nn.ModuleList([
            nn.Linear(hidden_dim, bd) for (_, bd) in cont_dims
        ])
        self.gamma_heads = nn.ModuleList([
            nn.Linear(hidden_dim, nd) for nd in cat_dims
        ])

    def forward(self, z, tau=0.2):
        h      = self.net(z)
        alphas = [torch.tanh(hd(h)) for hd in self.alpha_heads]
        betas  = [gumbel_softmax(hd(h), tau) for hd in self.beta_heads]
        gammas = [gumbel_softmax(hd(h), tau) for hd in self.gamma_heads]
        return alphas, betas, gammas


class TabVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 latent_dim, cont_dims, cat_dims):
        super().__init__()
        self.encoder   = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder   = Decoder(latent_dim, hidden_dim,
                                 cont_dims, cat_dims)
        self.log_delta = nn.Parameter(torch.zeros(len(cont_dims)))
        self.latent_dim = latent_dim

    def forward(self, x, tau=0.2):
        mu, log_var = self.encoder(x)
        z = Encoder.reparameterize(mu, log_var)
        alphas, betas, gammas = self.decoder(z, tau)
        return alphas, betas, gammas, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device='cpu', tau=0.2):
        z = torch.randn(n, self.latent_dim).to(device)
        return self.decoder(z, tau)


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
def tab_vae_loss(x_batch, alphas, betas, gammas,
                 mu, log_var,
                 cont_slices, cat_slices,
                 log_delta, beta_kl=1.0):
    recon = torch.tensor(0.0, device=x_batch.device)

    for i, (a_s, b_s, b_e) in enumerate(cont_slices):
        alpha_true = x_batch[:, a_s: a_s+1]
        beta_true  = x_batch[:, b_s: b_e]
        delta      = torch.exp(log_delta[i])
        recon += (0.5 * ((alphas[i] - alpha_true)**2) / delta).mean()
        if beta_true.shape[1] > 1:
            recon += F.cross_entropy(
                betas[i], beta_true.argmax(dim=1)
            )

    for i, (g_s, g_e) in enumerate(cat_slices):
        gamma_true = x_batch[:, g_s:g_e]
        if gamma_true.shape[1] > 1:
            recon += F.cross_entropy(
                gammas[i], gamma_true.argmax(dim=1)
            )

    kl = -0.5 * (
        1 + log_var - mu.pow(2) - log_var.exp()
    ).sum(dim=1).mean()

    return recon + beta_kl * kl, recon.item(), kl.item()


# ─────────────────────────────────────────────
# Train function
# ─────────────────────────────────────────────
def train_tab_vae(df: pd.DataFrame,
                  continuous_cols: list,
                  categorical_cols: list,
                  latent_dim=64,
                  hidden_dim=128,
                  epochs=100,
                  batch_size=256,
                  lr=1e-3,
                  beta_kl=1.0,
                  device='cpu') -> tuple:
    """
    Returns (trained_model, fitted_transformer,
             cont_slices, cat_slices)
    """
    transformer = TabTransformer(continuous_cols, categorical_cols)
    transformer.fit(df)
    X = transformer.transform(df)
    X_t = torch.tensor(X).to(device)

    cont_dims, cat_dims = transformer.get_decoder_info()

    # build slices for loss
    cont_slices, cat_slices = [], []
    ptr = 0
    for col in continuous_cols:
        info = transformer.col_info[col]
        n_m  = info['n_modes']
        cont_slices.append((ptr, ptr+1, ptr+1+n_m))
        ptr += 1 + n_m
    for col in categorical_cols:
        info = transformer.col_info[col]
        n_c  = info['n_cats']
        cat_slices.append((ptr, ptr+n_c))
        ptr += n_c

    model = TabVAE(X.shape[1], hidden_dim, latent_dim,
                   cont_dims, cat_dims).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True
    )

    tau_start, tau_end = 1.0, 0.2
    for epoch in range(epochs):
        tau = tau_start - (tau_start-tau_end) * epoch/epochs
        model.train()
        for (batch,) in loader:
            opt.zero_grad()
            alphas, betas, gammas, mu, lv = model(batch, tau)
            loss, _, _ = tab_vae_loss(
                batch, alphas, betas, gammas,
                mu, lv, cont_slices, cat_slices,
                model.log_delta, beta_kl
            )
            loss.backward()
            opt.step()

    model.eval()
    return model, transformer, cont_slices, cat_slices


# ─────────────────────────────────────────────
# Synthesize
# ─────────────────────────────────────────────
def synthesize(model: TabVAE,
               transformer: TabTransformer,
               n: int,
               continuous_cols: list,
               categorical_cols: list,
               device='cpu') -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        alphas, betas, gammas = model.sample(n, device=device)
    return transformer.inverse_transform(alphas, betas, gammas)