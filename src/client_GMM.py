# client_GMM.py
#
# Federated client using per-class Bayesian GMM for generative replay.
#
# Why GMM instead of a VAE?
#   - No gradient training, no KL collapse — EM convergence is guaranteed
#   - BayesianGaussianMixture automatically selects effective number of modes
#   - Per-class fitting: gmm_c.sample() always produces class-c windows
#     → labels are exact, not randomly drawn from a marginal distribution
#   - Lightweight: fits in seconds on CPU, no GPU needed
#   - Strong baseline: reveals how much VAE instability hurts vs. fundamental limits
#
# Architecture:
#   generators[domain_key] = {
#       class_label (int) : {
#           'gmm'   : BayesianGaussianMixture   (fitted, frozen after first domain step)
#           'n_obs' : int                        (training set size for this class)
#       }
#   }
#
# Data format (matches utils.load_data after reshape fix):
#   LSTM input : (B, T, F)   — T timesteps, F features per step
#   GMM input  : (N, T*F)    — flat windows stored/sampled, reshaped for LSTM
#
# Replay:
#   For each past domain, sample replay_per_domain windows proportionally
#   across classes, reshape to (B, T, F), feed into LSTM trainer.

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import utils
from evaluate_generator import evaluate_generator


class ClientGMM:

    def __init__(self, client_id, args, domain_path,
                 assigned_domains, device, model,
                 n_components: int = 10):
        """
        Parameters
        ----------
        n_components : maximum number of GMM components per class.
            BayesianGMM prunes unused components automatically, so setting
            this to a generous value (10–20) is safe — effective components
            will be fewer.
        """
        self.client_id        = client_id
        self.args             = args
        self.domain_path      = domain_path
        self.assigned_domains = assigned_domains
        self.device           = device
        self.n_components     = n_components

        # ── derived dims ─────────────────────────────────────────
        self.window_size    = self.args.window_size
        self.n_raw_features = self.args.n_raw_features

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
        # generators[domain_key] = {
        #     class_label: {'gmm': BayesianGaussianMixture, 'scaler': StandardScaler, 'n_obs': int}
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
    # PRIVATE — _train_generator()
    # ═══════════════════════════════════════════════════════════
    def _train_generator(self, domain_key: str):
        """
        Fit one BayesianGaussianMixture per class on the flat windows
        (N, T*F) from this domain's training data.

        BayesianGMM is chosen over plain GMM because:
          - It automatically prunes redundant components via the
            Dirichlet process weight prior (weight_concentration_prior)
          - No need to cross-validate n_components — set it large,
            let the model decide how many modes to keep
          - Closed-form EM: no risk of gradient instability
        """
        print(
            f"  [Client {self.client_id}] "
            f"Fitting GMM generators for domain: {domain_key} ..."
        )

        # ── collect all training batches ─────────────────────────
        all_X, all_y = [], []
        for data, target in self.train_domains_loader[domain_key]:
            # data: (B, T, F) → flatten to (B, T*F) for GMM
            flat = data.reshape(data.size(0), -1).cpu().numpy()
            all_X.append(flat)
            all_y.append(target.cpu().numpy())

        X_np = np.vstack(all_X)       # (N, T*F)
        y_np = np.concatenate(all_y)  # (N,)

        classes = np.unique(y_np)
        self.generators[domain_key] = {}

        for cls in classes:
            X_cls = X_np[y_np == cls]     # (N_c, T*F)
            n_cls = len(X_cls)

            # cap components to available samples to avoid GMM fitting errors
            n_comp = min(self.n_components, max(1, n_cls // 5))

            # ── standardize per class before fitting ──────────────
            # 'diag' covariance requires features at similar scales;
            # StandardScaler brings each dimension to zero-mean unit-variance.
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cls)

            print(
                f"  [Client {self.client_id}] "
                f"  Class {cls}: {n_cls} samples → "
                f"BayesianGMM(n_components={n_comp}, covariance_type='diag') ..."
            )

            # 'diag' covariance: one variance per dimension instead of a full
            # covariance matrix.  For T*F=140 dims, full needs ~9870 params/comp
            # (severely underdetermined with a few thousand samples); diag needs
            # only 140 params/comp → numerically stable and generalises better.
            gmm = BayesianGaussianMixture(
                n_components             = n_comp,
                covariance_type          = 'diag',
                weight_concentration_prior_type = 'dirichlet_process',
                weight_concentration_prior = 1e-2,   # sparse — prunes unused modes
                max_iter                 = 500,
                n_init                   = 3,
                random_state             = 42,
                reg_covar                = 1e-4,      # numerical stability
            )
            gmm.fit(X_scaled)

            effective = int((gmm.weights_ > 1e-3).sum())
            print(
                f"  [Client {self.client_id}] "
                f"  Class {cls}: GMM fitted ✓ "
                f"({effective}/{n_comp} effective components)"
            )

            self.generators[domain_key][int(cls)] = {
                'gmm'    : gmm,
                'scaler' : scaler,   # inverse-transform generated samples
                'n_obs'  : n_cls,
            }

        print(
            f"  [Client {self.client_id}] "
            f"GMM generators for {domain_key} frozen ✓ "
            f"| classes={list(self.generators[domain_key].keys())} "
            f"| total domains: {len(self.generators)}"
        )

        # ── evaluate generator quality ────────────────────────────
        syn_X_np, _ = self._synthesize_proportional(domain_key, n=1000)
        self.generator_eval_results[domain_key] = evaluate_generator(
            domain_key  = domain_key,
            real_X_np   = X_np,
            generator   = {},           # not used — syn_X_np provided directly
            n_synthetic = 1000,
            save_plots  = True,
            plots_dir   = 'results/gmm_plots',
            device      = str(self.device),
            syn_X_np    = syn_X_np,
        )

    # ═══════════════════════════════════════════════════════════
    # PRIVATE — _synthesize_proportional()
    # ═══════════════════════════════════════════════════════════
    def _synthesize_proportional(self, domain_key: str,
                                 n: int) -> tuple:
        """
        Sample n flat windows from GMM generators, preserving the
        real class distribution of this domain.

        For each class c:
            n_c = round(n * P(c))
            X_c, _ = gmm_c.sample(n_c)   ← always class-c samples
            y_c    = [c] * n_c            ← labels are exact, not drawn randomly

        Returns
        -------
        X_syn : (n_total, T*F)  float32  clipped to [0, 1]
        y_syn : (n_total,)      int64
        """
        gen_dict = self.generators[domain_key]
        total_obs = sum(v['n_obs'] for v in gen_dict.values())
        all_X, all_y = [], []

        for cls, entry in gen_dict.items():
            n_cls = max(1, round(n * entry['n_obs'] / total_obs))
            X_scaled, _ = entry['gmm'].sample(n_cls)
            # inverse-transform from standardized space back to original scale
            X_cls = entry['scaler'].inverse_transform(X_scaled)
            X_cls = np.clip(X_cls, 0.0, 1.0).astype(np.float32)
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
        Sample replay windows from all past GMM generators
        (excluding the current domain).

        Labels are deterministic — each GMM is class-specific.

        Returns a DataLoader with tensor shape (B, T, F) matching
        utils.load_data format. Returns None at the first time step.
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

            unique, counts = np.unique(y_syn, return_counts=True)
            print(
                f"  [Client {self.client_id}] "
                f"← Replayed {len(y_syn)} GMM samples from "
                f"domain {domain_key} "
                f"(classes: {dict(zip(unique.tolist(), counts.tolist()))})"
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
            f"GMM replay loader ready: "
            f"{len(y_all)} total samples from "
            f"{len(past_gens)} past domain(s) "
            f"| tensor shape: {tuple(X_tensor.shape)}"
        )
        return loader
