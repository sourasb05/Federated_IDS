# client_TVAE.py
#
# Drop-in replacement for client.py that adds per-domain
# generative replay using Tab-VAE.
#
# Data shape contract (from utils.load_data):
#   data:   (batch, seq_len=1, input_dim=window_size*raw_features)
#   target: (batch,)
#
# Generator strategy:
#   - One Tab-VAE per domain, trained once then frozen
#   - Generators are private to each client, never shared
#   - At each time step: train generator on current domain,
#     replay all past domains, train LSTM on combined data

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import utils
from tab_vae import train_tab_vae, synthesize


class ClientTabVAE:

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
            self.train_domains_loader[key], self.test_domains_loader[key] = utils.load_data(
                self.domain_path,
                key,
                files,
                window_size = self.args.window_size,
                step_size   = self.args.step_size,
                batch_size  = self.args.batch_size
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
        #   'model'      : TabVAE       (frozen),
        #   'transformer': TabTransformer,
        #   'cont_cols'  : list[str],
        #   'cat_cols'   : list[str],
        #   'label_dist' : pd.Series    empirical P(Y),
        #   'input_dim'  : int          flat feature dim
        # }
        self.generators: dict = {}

        # ── VAE hyperparameters ──────────────────
        # input_dim = window_size × raw_features
        # e.g. window_size=10, raw_features=14 → 140
        # latent_dim ~ input_dim / 4  (rule of thumb)
        self.vae_config = {
            'latent_dim' : 32,
            'hidden_dim' : 128,
            'epochs'     : 100,
            'batch_size' : 256,
            'lr'         : 1e-3,
            'beta_kl'    : 0.5,   # softer KL for high-dim input
        }

        # synthetic samples generated per past domain per round
        self.replay_per_domain = 500

    # ═══════════════════════════════════════════════════════
    # PUBLIC — train()
    # Identical structure to client.py + generator + replay
    # ═══════════════════════════════════════════════════════
    def train(self, global_model_state, time_step):
        """
        1. Load global weights into local model
        2. Train + freeze a Tab-VAE for current domain
           (only on the first global iteration it is seen)
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
            for data, target in self.train_domains_loader[current_domain]:
                # data:   (batch, 1, input_dim)
                # target: (batch,)
                data   = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output, _ = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            # ── synthetic replay data ────────────
            # same tensor shape: (batch, 1, input_dim)
            if replay_loader is not None:
                for data, target in replay_loader:
                    data   = data.to(self.device)
                    target = target.to(self.device)

                    self.optimizer.zero_grad()
                    output, _ = self.local_model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
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
        """
        Evaluate on local test set for current domain.
        Returns average loss. Identical to client.py.
        """
        self.eval_model.load_state_dict(model_state)
        self.eval_model.eval()

        total_loss     = 0.0
        current_domain = self.domain_keys[time_step]

        with torch.no_grad():
            for data, target in self.test_domains_loader[current_domain]:
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
        """
        Evaluate global model on local test data.
        Returns (loss, accuracy, f1). Identical to client.py.
        """
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
    # PRIVATE — _train_generator()
    # ═══════════════════════════════════════════════════════
    def _train_generator(self, domain_key: str):
        """
        Extract features from the existing DataLoader,
        train a Tab-VAE on them, then freeze it.

        Data shape from utils.load_data():
            data: (batch, seq_len=1, input_dim=140)

        We squeeze seq_len → (batch, 140) for the VAE,
        then unsqueeze back at synthesis time.
        """
        print(
            f"  [Client {self.client_id}] "
            f"Training Generator for domain: {domain_key} ..."
        )

        # ── collect all batches from DataLoader ──
        all_X, all_y = [], []
        for data, target in \
                self.train_domains_loader[domain_key]:
            # (batch, 1, input_dim) → (batch, input_dim)
            flat = data.squeeze(1).cpu().numpy()
            all_X.append(flat)
            all_y.append(target.cpu().numpy())

        X_np = np.vstack(all_X)       # (N, input_dim)
        y_np = np.concatenate(all_y)  # (N,)

        input_dim = X_np.shape[1]     # e.g. 140

        # ── build DataFrame ──────────────────────
        # data is already min-max normalised [0,1]
        # by utils.safe_minmax_normalize — treat all
        # columns as continuous, no categoricals
        col_names = [f'f_{i}' for i in range(input_dim)]
        df_X      = pd.DataFrame(X_np, columns=col_names)

        cont_cols = col_names
        cat_cols  = []

        # ── train Tab-VAE ────────────────────────
        vae_model, transformer, _, _ = train_tab_vae(
            df               = df_X,
            continuous_cols  = cont_cols,
            categorical_cols = cat_cols,
            latent_dim       = self.vae_config['latent_dim'],
            hidden_dim       = self.vae_config['hidden_dim'],
            epochs           = self.vae_config['epochs'],
            batch_size       = self.vae_config['batch_size'],
            lr               = self.vae_config['lr'],
            beta_kl          = self.vae_config['beta_kl'],
            device           = self.device
        )

        # ── freeze — never updated again ─────────
        for p in vae_model.parameters():
            p.requires_grad_(False)
        vae_model.eval()

        # ── store empirical P(Y) for this domain ─
        unique, counts = np.unique(y_np, return_counts=True)
        label_dist = pd.Series(
            counts / counts.sum(),
            index=unique
        )

        self.generators[domain_key] = {
            'model'      : vae_model,
            'transformer': transformer,
            'cont_cols'  : cont_cols,
            'cat_cols'   : cat_cols,
            'label_dist' : label_dist,
            'input_dim'  : input_dim,
        }

        print(
            f"  [Client {self.client_id}] "
            f"Generator_{domain_key}"
            f"| input_dim={input_dim} "
            f"| classes={label_dist.to_dict()} "
            f"| total generators: {len(self.generators)}"
        )

    # ═══════════════════════════════════════════════════════
    # PRIVATE — _build_replay_loader()
    # ═══════════════════════════════════════════════════════
    def _build_replay_loader(self, exclude_domain: str):
        """
        Synthesise windows from all past generators
        (excluding the current domain).

        Returns a DataLoader with the exact same tensor
        format as train_domains_loader:
            data:   (batch, seq_len=1, input_dim)
            target: (batch,)

        Returns None if there are no past generators yet
        (i.e. this is the very first domain).
        """
        past_gens = {
            k: v for k, v in self.generators.items()
            if k != exclude_domain
        }

        if not past_gens:
            return None

        all_X, all_y = [], []

        for domain_key, gen in past_gens.items():

            # ── synthesise flat features ─────────
            # returns DataFrame of shape (N, input_dim)
            syn_df = synthesize(
                model            = gen['model'],
                transformer      = gen['transformer'],
                n                = self.replay_per_domain,
                continuous_cols  = gen['cont_cols'],
                categorical_cols = gen['cat_cols'],
                device           = self.device
            )

            # clip to [0, 1] — matches min-max normalisation
            # applied by utils.safe_minmax_normalize()
            X_syn = np.clip(
                syn_df.values.astype(np.float32),
                0.0, 1.0
            )   # (replay_per_domain, input_dim)

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
                f"samples from Generator_{domain_key}"
            )

        X_all = np.vstack(all_X)       # (N_total, input_dim)
        y_all = np.concatenate(all_y)  # (N_total,)

        # unsqueeze seq_len=1 to match utils.load_data format:
        # (N, input_dim) → (N, 1, input_dim)
        X_tensor = torch.tensor(
            X_all, dtype=torch.float32
        ).unsqueeze(1)
        y_tensor = torch.tensor(y_all, dtype=torch.long)

        # sanity check against LSTM's expected input_size
        assert X_tensor.shape[2] == self.args.input_size, (
            f"[Client {self.client_id}] Shape mismatch: "
            f"VAE output dim={X_tensor.shape[2]} but "
            f"LSTM input_size={self.args.input_size}. "
            f"Check args.window_size × raw_feature_count."
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