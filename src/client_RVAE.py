# client_RVAE.py
#
# Federated client using CTVAE (Conditional Sequential VAE) for generative replay.
#
# Key improvements over the original RVAE client:
#   - Generator is CTVAE (class-conditional GRU encoder/decoder)
#   - Data is reshaped to (B, T, F) — LSTM sees real timesteps, not flat vectors
#   - Labels are GENERATED (conditioned decode), not sampled from P(Y) margin
#   - Per-class synthesis: attack windows look like attacks, benign like benign
#
# Everything else (FedAvg evaluation, server interface) is identical.

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import utils
from models import CTVAE, train_ctvae


class ClientRVAE:

    def __init__(self, client_id, args, domain_path,
                 assigned_domains, device, model):

        self.client_id        = client_id
        self.args             = args
        self.domain_path      = domain_path
        self.assigned_domains = assigned_domains
        self.device           = device

        # ── derived dims ─────────────────────────────────────────
        self.window_size    = self.args.window_size
        self.n_raw_features = self.args.n_raw_features   # set in main.py

        # ── models ───────────────────────────────────────────────
        self.local_model = copy.deepcopy(model).to(device)
        self.eval_model  = copy.deepcopy(model).to(device)
        self.local_loss_history = []

        # ── data loaders ─────────────────────────────────────────
        self.train_domains_loader = {}
        self.test_domains_loader  = {}

        domains = utils.create_domains(domain_path, assigned_domains)
        for key, files in domains.items():
            self.train_domains_loader[key], \
            self.test_domains_loader[key] = utils.load_data(
                self.domain_path,
                key,
                files,
                window_size    = self.window_size,
                step_size      = self.args.step_size,
                batch_size     = self.args.batch_size,
                n_raw_features = self.n_raw_features,
            )

        self.domain_keys = list(self.train_domains_loader.keys())

        # ── optimizer & loss ─────────────────────────────────────
        self.optimizer = optim.Adam(
            self.local_model.parameters(), lr=self.args.lr
        )
        self.criterion = nn.CrossEntropyLoss()

        # ── generator registry ────────────────────────────────────
        # key   : domain_key (str)
        # value : {
        #   'model'       : CTVAE  (frozen),
        #   'num_classes' : int,
        #   'class_counts': dict {label: count}  — for proportional synthesis
        # }
        self.generators: dict = {}

        # { domain_key -> evaluate_generator() result dict }
        self.generator_eval_results: dict = {}

        self.replay_per_domain = 500

    # ═══════════════════════════════════════════════════════════
    # PUBLIC — train()
    # ═══════════════════════════════════════════════════════════
    def train(self, global_model_state, time_step):
        self.local_model.load_state_dict(copy.deepcopy(global_model_state))
        self.local_model.train()

        current_domain = self.domain_keys[time_step]

        if current_domain not in self.generators:
            self._train_generator(current_domain)

        replay_loader = self._build_replay_loader(exclude_domain=current_domain)

        for epoch in range(self.args.local_epochs):
            epoch_loss = 0.0
            n_batches  = 0

            # ── real current-domain data ─────────────────────────
            for data, target in self.train_domains_loader[current_domain]:
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

            # ── synthetic replay data from past domains ───────────
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
            self.evaluate_local_model(
                self.local_model.state_dict(), time_step=time_step
            )

        return self.local_model.state_dict()

    # ═══════════════════════════════════════════════════════════
    # PUBLIC — evaluate_local_model()
    # ═══════════════════════════════════════════════════════════
    def evaluate_local_model(self, model_state, time_step):
        self.eval_model.load_state_dict(model_state)
        self.eval_model.eval()

        total_loss     = 0.0
        current_domain = self.domain_keys[time_step]

        with torch.no_grad():
            for data, target in self.test_domains_loader[current_domain]:
                data   = data.to(self.device)
                target = target.to(self.device)
                output, _ = self.eval_model(data)
                total_loss += self.criterion(output, target).item()

        return total_loss / len(self.test_domains_loader[current_domain])

    # ═══════════════════════════════════════════════════════════
    # PUBLIC — evaluate_global_model()
    # ═══════════════════════════════════════════════════════════
    def evaluate_global_model(self, model_state, time_step):
        self.eval_model.load_state_dict(model_state)
        self.eval_model.eval()

        total_loss  = 0.0
        all_preds   = []
        all_targets = []

        current_domain = self.domain_keys[time_step]

        with torch.no_grad():
            for data, target in self.test_domains_loader[current_domain]:
                data   = data.to(self.device)
                target = target.to(self.device)

                output, _ = self.eval_model(data)
                total_loss += self.criterion(output, target).item()

                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        evaluation_loss = total_loss / len(
            self.test_domains_loader[current_domain]
        )
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(
            all_targets, all_preds,
            average='macro', zero_division=0
        )

        print(
            f"  [Client {self.client_id}] "
            f"domain: {current_domain} "
            f"Eval Loss: {evaluation_loss:.4f}, "
            f"Acc: {accuracy:.4f}, "
            f"F1: {f1:.4f}"
        )
        return evaluation_loss, accuracy, f1

    # ═══════════════════════════════════════════════════════════
    # PRIVATE — _get_ctvae_config()
    # ═══════════════════════════════════════════════════════════
    def _get_ctvae_config(self, n_raw_features, n_samples):
        """
        Scale CTVAE hyperparameters to data dimensionality.

        hidden_dim : wide enough to capture per-step interactions
        latent_dim : small to keep KL and recon commensurable
                     After loss_factor=T×F scaling:
                       recon ≈ T×F × data_var
                       KL    ≈ 0.18 nats/dim × latent_dim
                     Target: latent_dim ≤ (T×F×var) / 0.18
        embed_dim  : compact class signal (8 dims is sufficient for 2 classes)
        """
        hidden_dim = max(128, n_raw_features * 8)
        latent_dim = max(8,   n_raw_features // 2)
        batch_size = min(256, max(32, n_samples // 20))

        return {
            'hidden_dim' : hidden_dim,
            'latent_dim' : latent_dim,
            'embed_dim'  : 8,
            'n_layers'   : 1,
            'epochs'     : 300,
            'free_bits'  : 1.0,
            'n_cycles'   : 4,
            'noise_std'  : 0.10,
            'batch_size' : batch_size,
            'lr'         : 1e-3,
        }

    # ═══════════════════════════════════════════════════════════
    # PRIVATE — _train_generator()
    # ═══════════════════════════════════════════════════════════
    def _train_generator(self, domain_key: str):
        print(
            f"  [Client {self.client_id}] "
            f"Training CTVAE Generator for domain: {domain_key} ..."
        )

        # ── collect all training batches ─────────────────────────
        all_X, all_y = [], []
        for data, target in self.train_domains_loader[domain_key]:
            # data is now (B, T, F) — flatten to (B, T*F) for storage
            flat = data.reshape(data.size(0), -1).cpu().numpy()
            all_X.append(flat)
            all_y.append(target.cpu().numpy())

        X_np = np.vstack(all_X)        # (N, T*F)
        y_np = np.concatenate(all_y)   # (N,)

        N_samples    = X_np.shape[0]
        num_classes  = int(y_np.max()) + 1

        cfg = self._get_ctvae_config(self.n_raw_features, N_samples)

        print(
            f"  [Client {self.client_id}] "
            f"CTVAE config: "
            f"hidden={cfg['hidden_dim']}  "
            f"latent={cfg['latent_dim']}  "
            f"embed={cfg['embed_dim']}  "
            f"free_bits={cfg['free_bits']}  "
            f"n_cycles={cfg['n_cycles']}  "
            f"noise_std={cfg['noise_std']}  "
            f"batch={cfg['batch_size']}"
        )

        ctvae_model = train_ctvae(
            X_np           = X_np,
            y_np           = y_np,
            window_size    = self.window_size,
            n_raw_features = self.n_raw_features,
            num_classes    = num_classes,
            hidden_dim     = cfg['hidden_dim'],
            latent_dim     = cfg['latent_dim'],
            embed_dim      = cfg['embed_dim'],
            n_layers       = cfg['n_layers'],
            epochs         = cfg['epochs'],
            batch_size     = cfg['batch_size'],
            lr             = cfg['lr'],
            noise_std      = cfg['noise_std'],
            free_bits      = cfg['free_bits'],
            n_cycles       = cfg['n_cycles'],
            device         = str(self.device),
        )

        # ── freeze ───────────────────────────────────────────────
        for p in ctvae_model.parameters():
            p.requires_grad_(False)
        ctvae_model.eval()

        # ── class counts for proportional synthesis ───────────────
        unique, counts = np.unique(y_np, return_counts=True)
        class_counts   = {int(k): int(v) for k, v in zip(unique, counts)}

        self.generators[domain_key] = {
            'model'       : ctvae_model,
            'num_classes' : num_classes,
            'class_counts': class_counts,
        }

        print(
            f"  [Client {self.client_id}] "
            f"CTVAE Generator_{domain_key} frozen ✓ "
            f"| window={self.window_size} × features={self.n_raw_features} "
            f"| classes={class_counts} "
            f"| total generators: {len(self.generators)}"
        )

        # ── evaluate generator quality ────────────────────────────
        syn_X_np, syn_y_np = self._synthesize_proportional(
            domain_key, n=1000
        )

        from evaluate_generator import evaluate_generator
        self.generator_eval_results[domain_key] = evaluate_generator(
            domain_key  = domain_key,
            real_X_np   = X_np,
            generator   = self.generators[domain_key],
            n_synthetic = 1000,
            save_plots  = True,
            plots_dir   = 'results/ctvae_plots',
            device      = str(self.device),
            syn_X_np    = syn_X_np,
            y_real      = y_np,
            y_syn       = syn_y_np,
        )

    # ═══════════════════════════════════════════════════════════
    # PRIVATE — _synthesize_proportional()
    # ═══════════════════════════════════════════════════════════
    def _synthesize_proportional(self, domain_key: str, n: int):
        """
        Generate n synthetic windows preserving the real class distribution.

        For each class c, generate floor(n * P(c)) samples conditioned on c.
        Labels are deterministic (not randomly sampled) — the decoder IS the label.

        Returns:
            X_syn : (n, T*F)  float32  in [0,1]
            y_syn : (n,)      int64
        """
        gen          = self.generators[domain_key]
        model        = gen['model']
        class_counts = gen['class_counts']

        total        = sum(class_counts.values())
        all_X, all_y = [], []

        for cls, cnt in class_counts.items():
            n_cls = max(1, round(n * cnt / total))
            # generate() returns (n_cls, T, F) numpy array
            X_cls = model.generate(
                n           = n_cls,
                class_label = cls,
                device      = str(self.device),
            )
            # reshape (n_cls, T, F) → (n_cls, T*F)
            X_cls = X_cls.reshape(n_cls, -1).astype(np.float32)
            X_cls = np.clip(X_cls, 0.0, 1.0)
            all_X.append(X_cls)
            all_y.append(np.full(n_cls, cls, dtype=np.int64))

        X_syn = np.vstack(all_X)
        y_syn = np.concatenate(all_y)
        return X_syn, y_syn

    # ═══════════════════════════════════════════════════════════
    # PRIVATE — _build_replay_loader()
    # ═══════════════════════════════════════════════════════════
    def _build_replay_loader(self, exclude_domain: str):
        """
        Synthesise windows from all past CTVAE generators
        (excluding the current domain).

        Labels come from the CTVAE decoder (class-conditioned),
        NOT from a random marginal draw. This preserves the
        decision boundary learned on past domains.

        Returns a DataLoader with tensor shape (B, T, F)
        matching utils.load_data format. Returns None if no
        past generators exist.
        """
        past_gens = {
            k: v for k, v in self.generators.items()
            if k != exclude_domain
        }

        if not past_gens:
            return None

        all_X, all_y = [], []

        for domain_key in past_gens:
            X_syn, y_syn = self._synthesize_proportional(
                domain_key, n=self.replay_per_domain
            )
            all_X.append(X_syn)
            all_y.append(y_syn)

            print(
                f"  [Client {self.client_id}] "
                f"← Replayed {len(y_syn)} class-conditioned "
                f"samples from CTVAE Generator_{domain_key} "
                f"(classes: {dict(zip(*np.unique(y_syn, return_counts=True)))})"
            )

        X_all = np.vstack(all_X)       # (N_total, T*F)
        y_all = np.concatenate(all_y)  # (N_total,)

        # (N, T*F) → (N, T, F) to match LSTM input format
        X_tensor = torch.tensor(
            X_all.reshape(-1, self.window_size, self.n_raw_features),
            dtype=torch.float32
        )
        y_tensor = torch.tensor(y_all, dtype=torch.long)

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
