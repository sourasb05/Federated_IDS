# client_TVAE.py
#
# Federated client using SimpleMSEVAE for generative replay.
#
# Generator pipeline:
#   1. Flatten windows (B, T, F) → (N, T*F)
#   2. PCA(0.99) → (N, pca_dim)   decorrelates features
#   3. SimpleMSEVAE trains on PCA components with MSE loss
#      (no MSN/GMM — PCA components are Gaussian, MSE is correct)
#   4. At replay: sample z ~ N(0,I) → decode → inverse-PCA → (N, T, F)
#
# Key hyperparameter fixes vs original TVAE:
#   - latent_dim = pca_dim // 2   (proper bottleneck, not overcomplete)
#   - free_bits KL               (no dimension collapse)
#   - No loss_factor > 1         (balanced ELBO)

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import utils
from tvae import train_simple_vae, synthesize_simple


class ClientTVAE:

    def __init__(self, client_id, args, domain_path,
                 assigned_domains, device, model):

        self.client_id        = client_id
        self.args             = args
        self.domain_path      = domain_path
        self.assigned_domains = assigned_domains
        self.device           = device

        # ── models — identical to client.py ─────
        self.local_model = copy.deepcopy(model).to(device)
        self.eval_model  = copy.deepcopy(model).to(device)
        self.local_loss_history = []

        # ── data loaders — identical to client.py ─
        self.train_domains_loader = {}
        self.test_domains_loader  = {}

        domains = utils.create_domains(domain_path, assigned_domains)
        for key, files in domains.items():
            self.train_domains_loader[key], \
            self.test_domains_loader[key] = utils.load_data(
                self.domain_path,
                key,
                files,
                window_size    = self.args.window_size,
                step_size      = self.args.step_size,
                batch_size     = self.args.batch_size,
                n_raw_features = getattr(self.args, 'n_raw_features', None),
            )

        self.domain_keys = list(self.train_domains_loader.keys())

        # ── optimizer & loss — identical to client.py
        self.optimizer = optim.Adam(
            self.local_model.parameters(), lr=self.args.lr
        )
        self.criterion = nn.CrossEntropyLoss()

        # ── generator registry ───────────────────
        # key   : domain_key (str)
        # value : {
        #   'model'      : SimpleMSEVAE  (frozen),
        #   'col_names'  : list[str]     PCA component names,
        #   'label_dist' : pd.Series     empirical P(Y),
        #   'input_dim'  : int           original T*F dim,
        #   'pca_dim'    : int,
        #   'pca'        : PCA           fitted, for inverse_transform
        # }
        self.generators: dict = {}

        # stores quality scores from evaluate_generator.py
        # { domain_key -> { 'quality': { overall, column_shapes, ... } } }
        self.generator_eval_results: dict = {}

        # vae_config is built dynamically in _get_vae_config(input_dim)
        # based on actual data dimensionality — see that method below
        self.vae_config: dict = {}

        # synthetic samples generated per past domain per round
        self.replay_per_domain = 500

    # ═══════════════════════════════════════════════════════
    # PUBLIC — train()
    # Identical structure to client.py + generator + replay
    # ═══════════════════════════════════════════════════════
    def train(self, global_model_state, time_step):
        """
        1. Load global weights into local model
        2. Train + freeze a TVAE for current domain
           (only on the first time this domain is seen)
        3. Build replay DataLoader from all past generators
        4. Train local LSTM on real data + replay data
        5. Return updated local state_dict
        """
        # step 1 — load global weights (identical to client.py)
        self.local_model.load_state_dict(
            copy.deepcopy(global_model_state)
        )
        self.local_model.train()

        current_domain = self.domain_keys[time_step]

        # step 2 — train generator once per domain
        if current_domain not in self.generators:
            self._train_generator(current_domain)

        # step 3 — replay loader from ALL past generators
        replay_loader = self._build_replay_loader(
            exclude_domain=current_domain
        )

        # step 4 — training loop (mirrors client.py exactly)
        for epoch in range(self.args.local_epochs):
            epoch_loss = 0.0
            n_batches  = 0

            # ── real current-domain data ─────────
            for data, target in \
                    self.train_domains_loader[current_domain]:
                # data:   (batch, seq_len=1, input_dim)
                # target: (batch,)
                data   = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output, _ = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.local_model.parameters(), max_norm=5.0
                )
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            # ── synthetic replay data ────────────
            # same tensor shape: (batch, seq_len=1, input_dim)
            if replay_loader is not None:
                for data, target in replay_loader:
                    data   = data.to(self.device)
                    target = target.to(self.device)

                    self.optimizer.zero_grad()
                    output, _ = self.local_model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.local_model.parameters(), max_norm=5.0
                    )
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.local_loss_history.append(avg_loss)

            # local eval after each epoch (identical to client.py)
            self.evaluate_local_model(
                self.local_model.state_dict(),
                time_step=time_step
            )

        return self.local_model.state_dict()

    # ═══════════════════════════════════════════════════════
    # PUBLIC — evaluate_local_model()
    # Identical to client.py
    # ═══════════════════════════════════════════════════════
    def evaluate_local_model(self, model_state, time_step):
        self.eval_model.load_state_dict(model_state)
        self.eval_model.eval()

        total_loss     = 0.0
        current_domain = self.domain_keys[time_step]

        with torch.no_grad():
            for data, target in \
                    self.test_domains_loader[current_domain]:
                data   = data.to(self.device)
                target = target.to(self.device)
                output, _ = self.eval_model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        evaluation_loss = total_loss / len(
            self.test_domains_loader[current_domain]
        )
        return evaluation_loss

    # ═══════════════════════════════════════════════════════
    # PUBLIC — evaluate_global_model()
    # Identical to client.py
    # ═══════════════════════════════════════════════════════
    def evaluate_global_model(self, model_state, time_step):
        self.eval_model.load_state_dict(model_state)
        self.eval_model.eval()

        total_loss  = 0.0
        all_preds   = []
        all_targets = []

        current_domain = self.domain_keys[time_step]

        with torch.no_grad():
            for data, target in \
                    self.test_domains_loader[current_domain]:
                data   = data.to(self.device)
                target = target.to(self.device)

                output, _ = self.eval_model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        evaluation_loss = total_loss / len(
            self.test_domains_loader[current_domain]
        )
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(
            all_targets, all_preds,
            average='macro',
            zero_division=0
        )

        print(
            f"  [Client {self.client_id}] "
            f"domain : {current_domain} "
            f"Eval Loss: {evaluation_loss:.4f}, "
            f"Acc: {accuracy:.4f}, "
            f"F1: {f1:.4f}"
        )
        return evaluation_loss, accuracy, f1

    # ═══════════════════════════════════════════════════════
    # PRIVATE — _get_vae_config()
    # ═══════════════════════════════════════════════════════
    def _get_vae_config(self, pca_dim: int, n_samples: int) -> dict:
        """
        Build SimpleMSEVAE config scaled to the PCA dimensionality.

        PCA components are Gaussian — MSN/GMM encoding is wrong for them.
        SimpleMSEVAE uses direct MSE on raw PCA values instead.

        SCALING RULES
        ─────────────
        hidden_dim : 4× pca_dim, floored at 64.
                     Wide enough to capture non-linear interactions
                     between components without over-fitting.

        latent_dim : pca_dim // 2, floored at 4, capped at 32.
                     Must be SMALLER than pca_dim to create real
                     compression pressure. Overcomplete bottleneck
                     (latent > input) kills generation quality.

        batch_size : min(256, max(32, n_samples // 20))
                     At least 20 gradient steps per epoch.

        free_bits  : 0.5 nats per dimension.
                     Prevents any latent dimension collapsing to the
                     prior (KL=0 → useless dimension at synthesis).
        """
        hidden_dim = max(64,  pca_dim * 4)
        latent_dim = max(4,   min(pca_dim // 2, 32))
        batch_size = min(256, max(32, n_samples // 20))

        return {
            'hidden_dim' : hidden_dim,
            'latent_dim' : latent_dim,
            'epochs'     : 300,
            'kl_warmup'  : 100,
            'batch_size' : batch_size,
            'lr'         : 1e-3,
            'free_bits'  : 0.5,
        }

    # ═══════════════════════════════════════════════════════
    # PRIVATE — _train_generator()
    # ═══════════════════════════════════════════════════════
    def _train_generator(self, domain_key: str):
        """
        Extract features from the existing DataLoader,
        train a TVAE on them, then freeze it.

        Data shape from utils.load_data():
            data: (batch, seq_len=1, input_dim=140)

        We squeeze seq_len → (batch, 140) for the TVAE.
        """
        print(
            f"  [Client {self.client_id}] "
            f"Training TVAE Generator for domain: {domain_key} ..."
        )

        # ── collect all batches ──────────────────
        all_X, all_y = [], []
        for data, target in \
                self.train_domains_loader[domain_key]:
            # (batch, T, F) → (batch, T*F)  [handles both T=1 and T>1]
            flat = data.reshape(data.size(0), -1).cpu().numpy()
            all_X.append(flat)
            all_y.append(target.cpu().numpy())

        X_np = np.vstack(all_X)       # (N, T*F)
        y_np = np.concatenate(all_y)  # (N,)

        input_dim = X_np.shape[1]     # e.g. T*F = 3*14 = 42
        N_samples = X_np.shape[0]     # total training rows

        # ── PCA decorrelation ─────────────────────
        # Decorrelates features so the VAE models independent
        # components. n_components=0.99 retains 99% of variance
        # and acts as mild noise reduction.
        from sklearn.decomposition import PCA

        pca = PCA(n_components=0.99, random_state=42)
        X_pca = pca.fit_transform(X_np)   # (N, pca_dim)

        pca_dim       = X_pca.shape[1]
        pca_explained = pca.explained_variance_ratio_.sum()
        col_names     = [f'pc_{i}' for i in range(pca_dim)]

        print(
            f"  [Client {self.client_id}] "
            f"PCA: {input_dim}d → {pca_dim}d "
            f"({pca_explained*100:.1f}% variance retained)"
        )

        # ── get config scaled to PCA dim ──────────
        cfg = self._get_vae_config(pca_dim, N_samples)

        print(
            f"  [Client {self.client_id}] "
            f"SimpleMSEVAE config for pca_dim={pca_dim}: "
            f"hidden={cfg['hidden_dim']} "
            f"latent={cfg['latent_dim']} "
            f"free_bits={cfg['free_bits']} "
            f"kl_warmup={cfg['kl_warmup']} "
            f"batch={cfg['batch_size']}"
        )

        # ── train SimpleMSEVAE on PCA-space data ──
        # Direct MSE on raw PCA values — no MSN/GMM transform.
        # PCA components are Gaussian so MSE is the correct loss.
        vae_model = train_simple_vae(
            X_np       = X_pca.astype(np.float32),
            hidden_dim = cfg['hidden_dim'],
            latent_dim = cfg['latent_dim'],
            epochs     = cfg['epochs'],
            batch_size = cfg['batch_size'],
            lr         = cfg['lr'],
            free_bits  = cfg['free_bits'],
            kl_warmup  = cfg['kl_warmup'],
            device     = str(self.device),
        )

        # ── freeze — never updated again ─────────
        for p in vae_model.parameters():
            p.requires_grad_(False)
        vae_model.eval()

        # ── store empirical P(Y) ─────────────────
        unique, counts = np.unique(y_np, return_counts=True)
        label_dist = pd.Series(
            counts / counts.sum(),
            index=unique
        )

        self.generators[domain_key] = {
            'model'      : vae_model,
            'col_names'  : col_names,   # PCA component names
            'label_dist' : label_dist,
            'input_dim'  : input_dim,   # original T*F dim
            'pca_dim'    : pca_dim,
            'pca'        : pca,         # fitted PCA for inverse_transform
        }

        print(
            f"  [Client {self.client_id}] "
            f"SimpleMSEVAE Generator_{domain_key} frozen ✓ "
            f"| input_dim={input_dim} → pca_dim={pca_dim} → latent={cfg['latent_dim']} "
            f"| classes={label_dist.to_dict()} "
            f"| total generators: {len(self.generators)}"
        )

        # ── synthesize 1000 samples for evaluation ────────────
        syn_pca_df = synthesize_simple(
            model     = vae_model,
            n         = 1000,
            col_names = col_names,
            device    = str(self.device),
        )
        X_pca_syn  = syn_pca_df.values.astype(np.float32)
        X_syn_eval = pca.inverse_transform(X_pca_syn)
        X_syn_eval = np.clip(X_syn_eval, 0.0, 1.0).astype(np.float32)

        # sample labels from empirical P(Y)
        y_syn_eval = np.random.choice(
            label_dist.index,
            size = len(X_syn_eval),
            p    = label_dist.values,
        ).astype(np.int64)

        from evaluate_generator import evaluate_generator

        # ── eval A: original feature space (T*F dims) ─────────
        # Cross-method comparison (TVAE vs RVAE).
        self.generator_eval_results[domain_key] = evaluate_generator(
            domain_key  = domain_key,
            real_X_np   = X_np,
            generator   = self.generators[domain_key],
            n_synthetic = 1000,
            save_plots  = True,
            plots_dir   = 'results/tvae_plots',
            device      = str(self.device),
            syn_X_np    = X_syn_eval,
            y_real      = y_np,
            y_syn       = y_syn_eval,
        )

        # ── eval B: PCA space (pca_dim dims) ──────────────────
        # Internal quality check: real PCA components vs VAE output.
        # No inverse-PCA smoothing — KS/Wasserstein reflect what
        # the VAE actually learned.
        self.generator_eval_results[domain_key + '_pca_space'] = evaluate_generator(
            domain_key  = domain_key + '_pca_space',
            real_X_np   = X_pca.astype(np.float32),
            generator   = self.generators[domain_key],
            n_synthetic = 1000,
            save_plots  = True,
            plots_dir   = 'results/tvae_plots_pca',
            device      = str(self.device),
            syn_X_np    = X_pca_syn,
            y_real      = y_np,
            y_syn       = y_syn_eval,
        )

    # ═══════════════════════════════════════════════════════
    # PRIVATE — _build_replay_loader()
    # ═══════════════════════════════════════════════════════
    def _build_replay_loader(self, exclude_domain: str):
        """
        Synthesise windows from all past generators
        (excluding the current domain).

        Returns a DataLoader with the exact same format
        as train_domains_loader:
            data:   (batch, T, F)
            target: (batch,)

        Returns None if no past generators exist.
        """
        past_gens = {
            k: v for k, v in self.generators.items()
            if k != exclude_domain
        }

        if not past_gens:
            return None

        all_X, all_y = [], []

        for domain_key, gen in past_gens.items():

            # synthesise in PCA space → DataFrame (N, pca_dim)
            syn_df = synthesize_simple(
                model     = gen['model'],
                n         = self.replay_per_domain,
                col_names = gen['col_names'],
                device    = self.device
            )

            # inverse PCA → restore original feature space (N, input_dim)
            X_pca_syn = syn_df.values.astype(np.float32)
            X_syn     = gen['pca'].inverse_transform(X_pca_syn)

            # clip to [0,1] — matches min-max normalisation
            X_syn = np.clip(X_syn, 0.0, 1.0).astype(np.float32)
            # shape: (replay_per_domain, input_dim)

            # sample labels from empirical P(Y)
            y_syn = np.random.choice(
                gen['label_dist'].index,
                size = self.replay_per_domain,
                p    = gen['label_dist'].values
            ).astype(np.int64)

            all_X.append(X_syn)
            all_y.append(y_syn)

            print(
                f"  [Client {self.client_id}] "
                f"← Replayed {self.replay_per_domain} "
                f"samples from TVAE Generator_{domain_key}"
            )

        X_all = np.vstack(all_X)       # (N_total, T*F)
        y_all = np.concatenate(all_y)  # (N_total,)

        # reshape (N, T*F) → (N, T, F) to match LSTM input format
        X_tensor = torch.tensor(
            X_all.reshape(-1, self.args.window_size,
                          getattr(self.args, 'n_raw_features', self.args.input_size)),
            dtype=torch.float32
        )
        y_tensor = torch.tensor(y_all, dtype=torch.long)

        # sanity check: last dim must equal n_raw_features (F), not T*F
        n_raw = getattr(self.args, 'n_raw_features',
                        self.args.input_size // self.args.window_size)
        assert X_tensor.shape[2] == n_raw, (
            f"[Client {self.client_id}] Shape mismatch: "
            f"TVAE output F={X_tensor.shape[2]} but "
            f"n_raw_features={n_raw}. "
            f"Check args.window_size × n_raw_features."
        )

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size = self.args.batch_size,
            shuffle    = True
        )

        print(
            f"  [Client {self.client_id}] "
            f"Replay loader ready: "
            f"{len(y_all)} total samples from "
            f"{len(past_gens)} past domain(s) "
            f"| tensor shape: {tuple(X_tensor.shape)}"
        )
        return loader