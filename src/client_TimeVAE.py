# client_TimeVAE.py
#
# Federated client that uses Base TimeVAE (Desai et al. 2021) for
# generative replay to mitigate catastrophic forgetting.
#
# Key difference from client_TVAE.py:
#   - No flattening, no PCA step.
#   - Generator trains directly on (N, T, F) windows — native temporal
#     structure is preserved by the Conv1D encoder/decoder.
#   - Replay samples come back as (B, T, F) tensors — no inverse transform.
#
# Generator pipeline:
#   1. Collect real windows from DataLoader: (B, T, F)
#   2. Train a TimeVAE on those windows (ELBO with KL warm-up)
#   3. Freeze the model
#   4. At replay: sample z ~ N(0,I) → decode → (B, T, F) → DataLoader
#
# Generator registry key schema:
#   {
#     'model'      : TimeVAE    (frozen)
#     'label_dist' : pd.Series  empirical P(Y)
#     'T'          : int        window_size
#     'F'          : int        n_raw_features
#   }

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import utils
from time_vae import train_time_vae, synthesize_time_vae
from evaluate_generator import evaluate_generator


class ClientTimeVAE:

    def __init__(self, client_id, args, domain_path,
                 assigned_domains, device, model):

        self.client_id        = client_id
        self.args             = args
        self.domain_path      = domain_path
        self.assigned_domains = assigned_domains
        self.device           = device

        # ── LSTM classifier (identical to Client) ─────
        self.local_model = copy.deepcopy(model).to(device)
        self.eval_model  = copy.deepcopy(model).to(device)
        self.local_loss_history = []

        # ── data loaders ──────────────────────────────
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

        # ── optimizer & loss ──────────────────────────
        self.optimizer = optim.Adam(
            self.local_model.parameters(), lr=self.args.lr
        )
        self.criterion = nn.CrossEntropyLoss()

        # ── generator registry ────────────────────────
        # key   : domain_key (str)
        # value : { 'model', 'label_dist', 'T', 'F' }
        self.generators: dict = {}

        # quality eval results (from evaluate_generator)
        self.generator_eval_results: dict = {}

        # synthetic samples generated per past domain per round
        self.replay_per_domain = 500

    # ═══════════════════════════════════════════════════════
    # PUBLIC — train()
    # ═══════════════════════════════════════════════════════
    def train(self, global_model_state, time_step):
        """
        1. Load global weights
        2. Train TimeVAE for current domain (once only)
        3. Build replay DataLoader from all past generators
        4. Train LSTM on real + replay data
        5. Return updated local state_dict
        """
        self.local_model.load_state_dict(
            copy.deepcopy(global_model_state)
        )
        self.local_model.train()

        current_domain = self.domain_keys[time_step]

        # step 2 — train generator once per domain
        if current_domain not in self.generators:
            self._train_generator(current_domain)

        # step 3 — build replay loader from ALL past generators
        replay_loader = self._build_replay_loader(
            exclude_domain=current_domain
        )

        # step 4 — training loop
        for epoch in range(self.args.local_epochs):
            epoch_loss = 0.0
            n_batches  = 0

            # real current-domain data
            for data, target in \
                    self.train_domains_loader[current_domain]:
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

            # synthetic replay data
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
                self.local_model.state_dict(),
                time_step=time_step
            )

        return self.local_model.state_dict()

    # ═══════════════════════════════════════════════════════
    # PUBLIC — evaluate_local_model()
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

        return total_loss / len(
            self.test_domains_loader[current_domain]
        )

    # ═══════════════════════════════════════════════════════
    # PUBLIC — evaluate_global_model()
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
            f"domain: {current_domain}  "
            f"Loss: {evaluation_loss:.4f}  "
            f"Acc: {accuracy:.4f}  "
            f"F1: {f1:.4f}"
        )
        return evaluation_loss, accuracy, f1

    # ═══════════════════════════════════════════════════════
    # PRIVATE — _get_timevae_config()
    # ═══════════════════════════════════════════════════════
    def _get_timevae_config(self, T: int, F: int, N: int) -> dict:
        """
        Build TimeVAE hyper-parameters scaled to window shape (T, F).

        latent_dim  : half of T*F, floored at 4, capped at 64.
                      Proper compression bottleneck — always < T*F.

        hidden_dim  : 4 * T * F, floored at 64.
                      Wide enough for the dense path (T < 5) to learn
                      all feature interactions without underfitting.

        recon_weight: scales with T*F so reconstruction gets equal
                      emphasis regardless of window size.
                      timevae_loss already normalises by T*F internally,
                      so this acts as a plain scalar multiplier (paper
                      range: 0.5–3.5). Use 2.0 to bias toward fidelity.

        free_bits   : 0.5 nats — prevents any latent dim collapsing
                      to the prior at synthesis time.
        """
        # T*F = total elements per window (e.g. 10*14 = 140)
        # latent_dim: ~10% of T*F — strong compression forces generalisation
        #             floor 4, cap 32 (not 64 — larger causes free_bits pinning)
        # free_bits : 0.5 / latent_dim keeps the TOTAL free_bits budget
        #             constant regardless of latent size, so KL is never
        #             fully pinned at the floor for all dims simultaneously
        latent_dim = max(4, min((T * F) // 10, 32))
        hidden_dim = max(64, T * F * 4)
        batch_size = min(256, max(32, N // 20))
        free_bits  = round(min(0.5, 8.0 / latent_dim), 4)  # total budget ~8 nats

        return {
            'latent_dim'  : latent_dim,
            'enc_filters' : (32, 64),
            'kernel_size' : 3,
            'hidden_dim'  : hidden_dim,
            'epochs'      : 300,
            'kl_warmup'   : 100,
            'batch_size'  : batch_size,
            'lr'          : 1e-3,
            'recon_weight': 2.0,
            'free_bits'   : free_bits,
        }

    # ═══════════════════════════════════════════════════════
    # PRIVATE — _train_generator()
    # ═══════════════════════════════════════════════════════
    def _train_generator(self, domain_key: str):
        """
        Collect (B, T, F) windows from the DataLoader,
        train a TimeVAE directly on them (no flattening / PCA),
        then freeze it.
        """
        print(
            f"  [Client {self.client_id}] "
            f"Training TimeVAE Generator for domain: {domain_key} ..."
        )

        # ── collect all batches ──────────────────
        all_X, all_y = [], []
        for data, target in self.train_domains_loader[domain_key]:
            # data: (B, T, F)  — already shaped by utils.load_data
            all_X.append(data.cpu().numpy())
            all_y.append(target.cpu().numpy())

        X_np = np.concatenate(all_X, axis=0).astype(np.float32)
        y_np = np.concatenate(all_y, axis=0)

        N, T, F = X_np.shape
        cfg     = self._get_timevae_config(T, F, N)

        print(
            f"  [Client {self.client_id}] "
            f"TimeVAE input verified: X_np.shape={X_np.shape} "
            f"(N={N}, T={T}, F={F})  min={X_np.min():.3f}  max={X_np.max():.3f}"
        )
        print(
            f"  [Client {self.client_id}] "
            f"TimeVAE config: latent={cfg['latent_dim']} "
            f"hidden={cfg['hidden_dim']} "
            f"filters={cfg['enc_filters']} "
            f"recon_w={cfg['recon_weight']} "
            f"free_bits={cfg['free_bits']} "
            f"batch={cfg['batch_size']}"
        )

        # ── train TimeVAE directly on (N, T, F) ──
        vae_model = train_time_vae(
            X_np         = X_np,
            T            = T,
            F            = F,
            latent_dim   = cfg['latent_dim'],
            enc_filters  = cfg['enc_filters'],
            kernel_size  = cfg['kernel_size'],
            hidden_dim   = cfg['hidden_dim'],
            epochs       = cfg['epochs'],
            batch_size   = cfg['batch_size'],
            lr           = cfg['lr'],
            recon_weight = cfg['recon_weight'],
            kl_warmup    = cfg['kl_warmup'],
            free_bits    = cfg['free_bits'],
            device       = str(self.device),
        )

        # ── freeze ────────────────────────────────
        for p in vae_model.parameters():
            p.requires_grad_(False)
        vae_model.eval()

        # ── empirical label distribution ──────────
        unique, counts = np.unique(y_np, return_counts=True)
        label_dist = pd.Series(
            counts / counts.sum(),
            index=unique
        )

        self.generators[domain_key] = {
            'model'      : vae_model,
            'label_dist' : label_dist,
            'T'          : T,
            'F'          : F,
        }

        print(
            f"  [Client {self.client_id}] "
            f"TimeVAE Generator_{domain_key} frozen ✓ "
            f"| shape=({T},{F}) → latent={cfg['latent_dim']} "
            f"| classes={label_dist.to_dict()} "
            f"| total generators: {len(self.generators)}"
        )

        # ── evaluate generator quality ─────────────────────────────
        # Synthesise 1000 windows, then compare distributions in the
        # ORIGINAL per-feature space (F dimensions, not T*F).
        #
        # Strategy: average over the T time steps within each window.
        #   real_eval  (N,    F) = X_np.mean(axis=1)
        #   syn_eval   (1000, F) = X_syn_3d.mean(axis=1)
        #
        # This collapses the temporal axis so we compare the per-feature
        # marginal distributions rather than every (timestep, feature)
        # combination.  The KS/Wasserstein stats are then interpretable
        # as "does feature k have the right distribution?" rather than
        # "does feature k at timestep t have the right distribution?".
        X_syn_3d = synthesize_time_vae(
            model  = vae_model,
            n      = 1000,
            device = str(self.device),
        )                                           # (1000, T, F)

        X_eval_real = X_np.mean(axis=1)             # (N,    F)
        X_eval_syn  = X_syn_3d.mean(axis=1)         # (1000, F)

        y_syn_eval = np.random.choice(
            label_dist.index,
            size = 1000,
            p    = label_dist.values,
        ).astype(np.int64)

        gen_proxy = {'label_dist': label_dist}

        self.generator_eval_results[domain_key] = evaluate_generator(
            domain_key  = domain_key,
            real_X_np   = X_eval_real,
            generator   = gen_proxy,
            n_synthetic = 1000,
            save_plots  = True,
            plots_dir   = 'results/timevae_plots',
            device      = str(self.device),
            syn_X_np    = X_eval_syn,
            y_real      = y_np,
            y_syn       = y_syn_eval,
        )

    # ═══════════════════════════════════════════════════════
    # PRIVATE — _build_replay_loader()
    # ═══════════════════════════════════════════════════════
    def _build_replay_loader(self, exclude_domain: str):
        """
        Synthesise (T, F) windows from all past generators,
        sample labels from empirical P(Y), and return a DataLoader
        with the same (B, T, F) format as train_domains_loader.

        Returns None when no past generators exist.
        """
        past_gens = {
            k: v for k, v in self.generators.items()
            if k != exclude_domain
        }

        if not past_gens:
            return None

        all_X, all_y = [], []

        for domain_key, gen in past_gens.items():
            # synthesise → (replay_per_domain, T, F)
            X_syn = synthesize_time_vae(
                model  = gen['model'],
                n      = self.replay_per_domain,
                device = self.device,
            )

            # sample labels from empirical P(Y)
            y_syn = np.random.choice(
                gen['label_dist'].index,
                size = self.replay_per_domain,
                p    = gen['label_dist'].values,
            ).astype(np.int64)

            all_X.append(X_syn)
            all_y.append(y_syn)

            print(
                f"  [Client {self.client_id}] "
                f"← Replayed {self.replay_per_domain} "
                f"samples from TimeVAE Generator_{domain_key}"
            )

        X_all = np.concatenate(all_X, axis=0)  # (N_total, T, F)
        y_all = np.concatenate(all_y, axis=0)  # (N_total,)

        X_tensor = torch.tensor(X_all, dtype=torch.float32)
        y_tensor = torch.tensor(y_all, dtype=torch.long)

        # sanity check: shape must match LSTM input
        T_expected = self.args.window_size
        F_expected = getattr(self.args, 'n_raw_features',
                             self.args.input_size // self.args.window_size)
        assert X_tensor.shape[1] == T_expected and \
               X_tensor.shape[2] == F_expected, (
            f"[Client {self.client_id}] Replay shape mismatch: "
            f"got {tuple(X_tensor.shape[1:])} "
            f"expected ({T_expected}, {F_expected}). "
            f"Check args.window_size and n_raw_features."
        )

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size = self.args.batch_size,
            shuffle    = True,
        )

        print(
            f"  [Client {self.client_id}] "
            f"Replay loader ready: "
            f"{len(y_all)} total samples from "
            f"{len(past_gens)} past domain(s) "
            f"| tensor shape: {tuple(X_tensor.shape)}"
        )
        return loader
