# zivae_batch.py
#
# Runs the ZI-VAE evaluation on every CSV in every attack_data domain.
#
# Output structure (mirrors attack_data layout):
#   toy/zivae_results/
#     <domain>/
#       <csv_stem>/
#         distributions.png
#         ks_bar.png
#         correlations.png
#         metrics.json          ← KS stats + TSTR/TRTS per file
#     summary.csv               ← one row per CSV across all domains
#
# Run:  python toy/zivae_batch.py
#       python toy/zivae_batch.py --domain blackhole_var10_base   (single domain)
#       python toy/zivae_batch.py --workers 4                     (parallel)

from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

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

# ── paths ──────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
SRC         = REPO_ROOT / 'src'
ATTACK_DATA = REPO_ROOT / 'attack_data'
RESULTS_DIR = Path(__file__).resolve().parent / 'zivae_results'

sys.path.insert(0, str(SRC))
from time_vae import train_time_vae, synthesize_time_vae

# ── global config ──────────────────────────────────────────────
WINDOW_SIZE   = 10
N_SYNTH       = 500     # per class (kept smaller for speed across 960 files)
EPOCHS        = 200
DEVICE        = 'cpu'
SPARSE_THRESH = 0.30


# ═══════════════════════════════════════════════════════════════
# ZI-VAE — architecture (same as zivae_toy.py)
# ═══════════════════════════════════════════════════════════════

class ZIVAEEncoder(nn.Module):
    def __init__(self, T, F, latent_dim, hidden_dim):
        super().__init__()
        self.net    = nn.Sequential(
            nn.Linear(T * F, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x.flatten(1))
        return self.fc_mu(h), self.fc_var(h)


class ZIVAEDecoder(nn.Module):
    def __init__(self, T, F, latent_dim, hidden_dim, sparse_idx):
        super().__init__()
        self.T          = T
        self.F          = F
        self.sparse_idx = sparse_idx
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, T * F),
        )
        if sparse_idx:
            self.gate_net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, T * len(sparse_idx)),
            )
        else:
            self.gate_net = None

    def forward(self, z):
        B          = z.size(0)
        gauss_out  = torch.sigmoid(self.net(z)).view(B, self.T, self.F)
        gate_logit = (self.gate_net(z).view(B, self.T, len(self.sparse_idx))
                      if self.gate_net is not None else None)
        return gauss_out, gate_logit


class ZIVAE(nn.Module):
    def __init__(self, T, F, latent_dim, hidden_dim, sparse_idx):
        super().__init__()
        self.T          = T
        self.F          = F
        self.latent_dim = latent_dim
        self.sparse_idx = sparse_idx
        self.encoder    = ZIVAEEncoder(T, F, latent_dim, hidden_dim)
        self.decoder    = ZIVAEDecoder(T, F, latent_dim, hidden_dim, sparse_idx)

    @staticmethod
    def _reparam(mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def forward(self, x):
        mu, lv        = self.encoder(x)
        z             = self._reparam(mu, lv)
        gauss, gate   = self.decoder(z)
        return gauss, gate, mu, lv

    @torch.no_grad()
    def sample(self, n, device='cpu'):
        z             = torch.randn(n, self.latent_dim, device=device)
        gauss, gate   = self.decoder(z)
        out           = gauss.clone()
        if gate is not None:
            g = torch.bernoulli(torch.sigmoid(gate)).bool()
            for s_local, s_global in enumerate(self.sparse_idx):
                out[:, :, s_global] = torch.where(
                    g[:, :, s_local],
                    gauss[:, :, s_global],
                    torch.zeros_like(gauss[:, :, s_global])
                )
        return out.cpu().numpy().astype(np.float32)


def _zivae_loss(x, gauss, gate, mu, lv, sparse_idx,
                recon_w=2.0, kl_w=1.0, free_bits=0.5):
    B   = x.shape[0]
    TF  = x.shape[1] * x.shape[2]
    mse = F_torch.mse_loss(gauss, x, reduction='sum') / B
    bce = torch.tensor(0.0, device=x.device)
    if gate is not None:
        for sl, sg in enumerate(sparse_idx):
            lbl = (x[:, :, sg] > 0).float()
            bce = bce + F_torch.binary_cross_entropy_with_logits(
                gate[:, :, sl], lbl, reduction='sum') / B
    kl_pd = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
    kl_pd = kl_pd.clamp(min=free_bits)
    kl    = kl_pd.sum(1).mean()
    total = (recon_w / TF) * mse + 0.1 * bce + kl_w * kl
    return total, mse.item() / TF, kl.item()


def _train_zivae(X_np, T, F, sparse_idx, latent_dim, hidden_dim,
                 epochs, batch_size, free_bits, device='cpu'):
    X_t    = torch.tensor(X_np, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size, shuffle=True,
    )
    model  = ZIVAE(T, F, latent_dim, hidden_dim, sparse_idx).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    warmup = max(100, 1)

    model.train()
    for epoch in range(epochs):
        kl_w = min(1.0, (epoch + 1) / warmup)
        for (b,) in loader:
            opt.zero_grad()
            gauss, gate, mu, lv = model(b)
            loss, _, _ = _zivae_loss(b, gauss, gate, mu, lv, sparse_idx,
                                     kl_w=kl_w, free_bits=free_bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════

def load_and_window(csv_path, window_size):
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    feat_cols = [c for c in df.columns if c != 'label']

    sparse_cols = [
        c for c in feat_cols
        if (pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]))
        and (df[c] == 0).mean() > SPARSE_THRESH
    ]
    sparse_idx = [feat_cols.index(c) for c in sparse_cols]

    split  = int(len(df) * 0.8)
    tr, te = df.iloc[:split].copy(), df.iloc[split:].copy()
    g_min  = tr[feat_cols].min()
    g_max  = tr[feat_cols].max()
    denom  = (g_max - g_min).replace(0, 1)

    for d in [tr, te]:
        d[feat_cols] = ((d[feat_cols] - g_min) / denom).clip(0, 1).fillna(0)

    def _win(d):
        X, y = [], []
        vals   = d[feat_cols].values.astype(np.float32)
        labels = d['label'].values.astype(int)
        for i in range(len(vals) - window_size):
            X.append(vals[i:i + window_size])
            y.append(labels[i + window_size - 1])
        return np.array(X, np.float32), np.array(y, np.int64)

    X_tr, y_tr = _win(tr)
    X_te, y_te = _win(te)
    return X_tr, y_tr, X_te, y_te, feat_cols, sparse_idx


def clf_score(X_tr, y_tr, X_te, y_te):
    try:
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                  random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        return (float(accuracy_score(y_te, preds)),
                float(f1_score(y_te, preds, average='macro', zero_division=0)))
    except Exception:
        return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════
# SINGLE-FILE EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_file(csv_path: Path, out_dir: Path) -> dict:
    """
    Run ZI-VAE on one CSV.  Returns a metrics dict and saves plots to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────
    try:
        X_tr, y_tr, X_te, y_te, feat_cols, sparse_idx = load_and_window(
            str(csv_path), WINDOW_SIZE
        )
    except Exception as e:
        return {'error': str(e), 'csv': str(csv_path)}

    N, T, F = X_tr.shape
    classes = sorted(np.unique(y_tr).tolist())

    if N < 20:
        return {'error': 'too few windows', 'csv': str(csv_path)}

    # ── hyper-params ──────────────────────────────────────────
    latent_dim = max(4, min((T * F) // 10, 32))
    hidden_dim = max(64, T * F * 4)
    free_bits  = round(min(0.5, 8.0 / latent_dim), 4)
    batch_size = min(256, max(32, N // 20))

    X_tr_flat = X_tr.mean(axis=1)
    X_te_flat = X_te.mean(axis=1)

    # ── train one ZI-VAE per class ────────────────────────────
    syn_X, syn_y = [], []
    for cls in classes:
        X_cls = X_tr[y_tr == cls]
        if len(X_cls) < batch_size:
            continue
        model = _train_zivae(
            X_np       = X_cls,
            T=T, F=F,
            sparse_idx = sparse_idx,
            latent_dim = latent_dim,
            hidden_dim = hidden_dim,
            epochs     = EPOCHS,
            batch_size = batch_size,
            free_bits  = free_bits,
            device     = DEVICE,
        )
        X_syn = model.sample(N_SYNTH, device=DEVICE)
        syn_X.append(X_syn)
        syn_y.append(np.full(N_SYNTH, cls, dtype=np.int64))

    if not syn_X:
        return {'error': 'all classes skipped', 'csv': str(csv_path)}

    X_syn_all  = np.concatenate(syn_X, axis=0)
    y_syn_all  = np.concatenate(syn_y, axis=0)
    X_syn_flat = X_syn_all.mean(axis=1)

    # ── KS ────────────────────────────────────────────────────
    ks_rows = []
    for i, col in enumerate(feat_cols):
        s, p = stats.ks_2samp(X_tr_flat[:, i], X_syn_flat[:, i])
        ks_rows.append({'feature': col, 'ks_stat': round(float(s), 4),
                        'p_value': round(float(p), 4),
                        'pass': bool(p > 0.05)})
    ks_df = pd.DataFrame(ks_rows).sort_values('ks_stat', ascending=False)

    # ── TSTR / TRTS / Baseline ────────────────────────────────
    _, tstr_f1 = clf_score(X_syn_flat, y_syn_all, X_te_flat,  y_te)
    _, trts_f1 = clf_score(X_tr_flat,  y_tr,      X_syn_flat, y_syn_all)
    _, base_f1 = clf_score(X_tr_flat,  y_tr,      X_te_flat,  y_te)

    # ── plots ─────────────────────────────────────────────────
    _plot_distributions(X_tr_flat, X_syn_flat, feat_cols, sparse_idx,
                        ks_df, out_dir)
    _plot_ks_bar(ks_df, out_dir)
    _plot_correlations(X_tr_flat, X_syn_flat, feat_cols, out_dir)

    # ── metrics dict ──────────────────────────────────────────
    metrics = {
        'csv'        : str(csv_path),
        'domain'     : csv_path.parent.name,
        'file'       : csv_path.name,
        'n_train'    : int(N),
        'n_features' : int(F),
        'n_sparse'   : len(sparse_idx),
        'sparse_features': [feat_cols[i] for i in sparse_idx],
        'ks_mean'    : round(float(ks_df['ks_stat'].mean()), 4),
        'ks_pass'    : int(ks_df['pass'].sum()),
        'ks_total'   : int(len(ks_df)),
        'ks_per_feature': ks_df.to_dict(orient='records'),
        'tstr_f1'    : round(tstr_f1, 4),
        'trts_f1'    : round(trts_f1, 4),
        'base_f1'    : round(base_f1, 4),
        'gap'        : round(abs(tstr_f1 - trts_f1), 4),
    }

    with open(out_dir / 'metrics.json', 'w') as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


# ═══════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════

def _plot_distributions(real_flat, syn_flat, feat_cols, sparse_idx, ks_df, out_dir):
    n_feat  = len(feat_cols)
    ks_lu   = ks_df.set_index('feature')
    real_df = pd.DataFrame(real_flat, columns=feat_cols)
    syn_df  = pd.DataFrame(syn_flat,  columns=feat_cols)

    cols_per_row = min(n_feat, 7)
    n_rows       = (n_feat + cols_per_row - 1) // cols_per_row
    fig, axes    = plt.subplots(n_rows, cols_per_row,
                                figsize=(2.8 * cols_per_row, 3.2 * n_rows),
                                constrained_layout=True)
    axes = np.array(axes).flatten()

    for idx, col in enumerate(feat_cols):
        ax  = axes[idx]
        r   = real_df[col].values
        s   = syn_df[col].values
        ax.hist(r, bins=20, alpha=0.4, color='steelblue', density=True, label='Real')
        ax.hist(s, bins=20, alpha=0.4, color='mediumpurple', density=True, label='ZI-VAE')
        try:
            xs = np.linspace(min(r.min(), s.min()), max(r.max(), s.max()), 100)
            ax.plot(xs, stats.gaussian_kde(r)(xs), color='steelblue', lw=1.0)
            ax.plot(xs, stats.gaussian_kde(s)(xs), color='mediumpurple', lw=1.0)
        except Exception:
            pass
        ks_v = ks_lu.loc[col, 'ks_stat'] if col in ks_lu.index else float('nan')
        star = '*' if idx in sparse_idx else ''
        ax.set_title(f"{star}{col}\nKS={ks_v:.3f}", fontsize=6)
        ax.tick_params(labelsize=5)
        if idx == 0:
            ax.legend(fontsize=5)

    for ax in axes[len(feat_cols):]:
        ax.set_visible(False)

    fig.suptitle('ZI-VAE — Real vs Synthetic Distributions  (* = sparse)',
                 fontsize=9)
    fig.savefig(out_dir / 'distributions.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def _plot_ks_bar(ks_df, out_dir):
    fig, ax = plt.subplots(figsize=(max(8, len(ks_df) * 0.7), 4),
                           constrained_layout=True)
    df_s = ks_df.sort_values('ks_stat', ascending=False)
    bar_colors = ['steelblue' if p else 'tomato' for p in df_s['pass'].tolist()]
    ax.bar(df_s['feature'], df_s['ks_stat'], color=bar_colors)
    ax.axhline(0.05, color='black', linestyle='--', lw=1.2, label='p=0.05 threshold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('KS Statistic')
    ax.set_title(f"KS per Feature — mean={ks_df['ks_stat'].mean():.3f}  "
                 f"pass={ks_df['pass'].sum()}/{len(ks_df)}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / 'ks_bar.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def _plot_correlations(real_flat, syn_flat, feat_cols, out_dir):
    real_df = pd.DataFrame(real_flat, columns=feat_cols)
    syn_df  = pd.DataFrame(syn_flat,  columns=feat_cols)
    rc, sc  = real_df.corr(), syn_df.corr()
    dc      = (rc - sc).abs()
    kw      = dict(cmap='coolwarm', vmin=-1, vmax=1, square=True,
                   annot=True, fmt='.2f', annot_kws={'size': 5},
                   xticklabels=True, yticklabels=True)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5),
                                      constrained_layout=True)
    sns.heatmap(rc, ax=a1, **kw);  a1.set_title('Real')
    sns.heatmap(sc, ax=a2, **kw);  a2.set_title('Synthetic (ZI-VAE)')
    sns.heatmap(dc, ax=a3, cmap='Reds', vmin=0, vmax=1, square=True,
                annot=True, fmt='.2f', annot_kws={'size': 5},
                xticklabels=True, yticklabels=True)
    a3.set_title('|Difference|')
    fig.suptitle('Correlation Matrices', fontsize=10)
    fig.savefig(out_dir / 'correlations.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# BATCH RUNNER
# ═══════════════════════════════════════════════════════════════

def collect_csvs(attack_data_root: Path, domain_filter: str | None):
    """Return list of (domain_name, csv_path) sorted."""
    pairs = []
    for domain_dir in sorted(attack_data_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        if domain_filter and domain_dir.name != domain_filter:
            continue
        for csv_path in sorted(domain_dir.glob('*.csv')):
            pairs.append((domain_dir.name, csv_path))
    return pairs


def run_batch(domain_filter=None, n_workers=1):
    pairs = collect_csvs(ATTACK_DATA, domain_filter)
    total = len(pairs)
    print(f"\nZI-VAE batch evaluation")
    print(f"  CSVs to process : {total}")
    print(f"  Results root    : {RESULTS_DIR}")
    print(f"  Window size     : {WINDOW_SIZE}")
    print(f"  Epochs          : {EPOCHS}")
    print(f"  N_synth/class   : {N_SYNTH}\n")

    all_metrics = []
    failed      = []

    for i, (domain, csv_path) in enumerate(pairs, 1):
        stem    = csv_path.stem
        out_dir = RESULTS_DIR / domain / stem
        print(f"[{i:>4}/{total}] {domain}/{csv_path.name} ", end='', flush=True)

        try:
            m = evaluate_file(csv_path, out_dir)
            if 'error' in m:
                print(f"SKIP  ({m['error']})")
                failed.append({'domain': domain, 'file': csv_path.name,
                               'reason': m['error']})
            else:
                print(f"KS={m['ks_mean']:.3f} pass={m['ks_pass']}/{m['ks_total']}  "
                      f"TSTR={m['tstr_f1']:.3f}  TRTS={m['trts_f1']:.3f}")
                all_metrics.append(m)
        except Exception:
            msg = traceback.format_exc().splitlines()[-1]
            print(f"ERROR  {msg}")
            failed.append({'domain': domain, 'file': csv_path.name,
                           'reason': msg})

    # ── summary CSV ───────────────────────────────────────────
    if all_metrics:
        summary_cols = ['domain', 'file', 'n_train', 'n_features',
                        'n_sparse', 'ks_mean', 'ks_pass', 'ks_total',
                        'tstr_f1', 'trts_f1', 'base_f1', 'gap']
        summary_df = pd.DataFrame([
            {k: m[k] for k in summary_cols} for m in all_metrics
        ])
        summary_path = RESULTS_DIR / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n── Summary ────────────────────────────────────────")
        print(f"  Processed : {len(all_metrics)}")
        print(f"  Skipped   : {len(failed)}")
        print(f"  Mean KS   : {summary_df['ks_mean'].mean():.4f}")
        print(f"  Mean TSTR : {summary_df['tstr_f1'].mean():.4f}")
        print(f"  Mean TRTS : {summary_df['trts_f1'].mean():.4f}")
        print(f"  Saved     : {summary_path}")

        # per-domain summary
        dom_summary = summary_df.groupby('domain')[
            ['ks_mean', 'ks_pass', 'tstr_f1', 'trts_f1']
        ].mean().round(4)
        print(f"\n── Per-domain averages ────────────────────────────")
        print(dom_summary.to_string())

    if failed:
        fail_path = RESULTS_DIR / 'failed.csv'
        pd.DataFrame(failed).to_csv(fail_path, index=False)
        print(f"\n  Failed files saved to: {fail_path}")

    print(f"\nDone. Results at: {RESULTS_DIR}\n")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ZI-VAE batch evaluation over all attack_data domains'
    )
    parser.add_argument('--domain', default=None,
                        help='Run only this domain (e.g. blackhole_var10_base)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_batch(domain_filter=args.domain)
