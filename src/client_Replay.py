# client_Replay.py
#
# Federated client using Experience Replay (stored real samples).
#
# Instead of training a generative model, this client keeps a fixed
# buffer of REAL samples drawn from each domain's training data.
# During training on a new domain, past-domain buffers are replayed
# exactly as if they were generated — but they are real data, so the
# KS test trivially passes (same underlying distribution by construction).
#
# Key differences from ClientRVAE / ClientTVAE:
#   - No generator, no VAE, no PCA — just a dict of stored numpy arrays
#   - _store_buffer()       replaces _train_generator()
#   - _build_replay_loader() draws directly from stored buffers
#   - evaluate_buffer()     calls evaluate_generator(syn_X_np=buffer)
#     → KS p >> 0.05 since buffer IS a sample from the real distribution
#
# Everything else (LSTM training, FedAvg evaluation) is identical
# to client_RVAE.py.

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import utils


class ClientReplay:

    def __init__(self, client_id, args, domain_path,
                 assigned_domains, device, model,
                 buffer_size: int = 500):
        """
        Args:
            buffer_size : number of real samples stored per domain.
                          Larger buffer → better coverage; smaller →
                          lower memory footprint and faster replay.
        """
        self.client_id        = client_id
        self.args             = args
        self.domain_path      = domain_path
        self.assigned_domains = assigned_domains
        self.device           = device
        self.buffer_size      = buffer_size

        # ── models ───────────────────────────────
        self.local_model = copy.deepcopy(model).to(device)
        self.eval_model  = copy.deepcopy(model).to(device)
        self.local_loss_history = []

        # ── data loaders ─────────────────────────
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

        # ── optimizer & loss ─────────────────────
        self.optimizer = optim.Adam(
            self.local_model.parameters(), lr=self.args.lr
        )
        self.criterion = nn.CrossEntropyLoss()

        # ── derived dims ─────────────────────────
        self.window_size    = self.args.window_size
        self.n_raw_features = getattr(self.args, 'n_raw_features',
                                      self.args.input_size // self.args.window_size)

        # ── experience replay buffers ─────────────
        # key   : domain_key (str)
        # value : {
        #   'X'         : np.ndarray  (buffer_size, input_dim)  real samples
        #   'label_dist': pd.Series   empirical P(Y)
        # }
        self.buffers: dict = {}

        # { domain_key -> evaluate_generator() result dict }
        self.buffer_eval_results: dict = {}

    # ═══════════════════════════════════════════════════════
    # PUBLIC — train()
    # ═══════════════════════════════════════════════════════
    def train(self, global_model_state, time_step):
        self.local_model.load_state_dict(copy.deepcopy(global_model_state))
        self.local_model.train()

        current_domain = self.domain_keys[time_step]

        # Store buffer for current domain (once, then frozen)
        if current_domain not in self.buffers:
            self._store_buffer(current_domain)

        replay_loader = self._build_replay_loader(
            exclude_domain=current_domain
        )

        for epoch in range(self.args.local_epochs):
            epoch_loss = 0.0
            n_batches  = 0

            # ── real current-domain data ─────────
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

            # ── real replay data from past domains ───
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

    # ═══════════════════════════════════════════════════════
    # PUBLIC — evaluate_local_model()
    # ═══════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════
    # PRIVATE — _store_buffer()
    # ═══════════════════════════════════════════════════════
    def _store_buffer(self, domain_key: str):
        """
        Randomly sample buffer_size real windows from the training loader
        and store them in self.buffers[domain_key].

        The buffer contains REAL data, so KS tests between the buffer
        and the full training set will give p >> 0.05 (both samples
        are drawn from the same underlying distribution).
        """
        print(
            f"  [Client {self.client_id}] "
            f"Building Replay Buffer for domain: {domain_key} ..."
        )

        # ── collect all training samples ──────────
        # data is now (B, T, F); flatten to (B, T*F) for buffer storage
        all_X, all_y = [], []
        for data, target in self.train_domains_loader[domain_key]:
            flat = data.reshape(data.size(0), -1).cpu().numpy()  # (B, T*F)
            all_X.append(flat)
            all_y.append(target.cpu().numpy())

        X_np = np.vstack(all_X)       # (N, T*F)
        y_np = np.concatenate(all_y)  # (N,)

        N_samples = X_np.shape[0]

        # ── random subsample ──────────────────────
        n_store = min(self.buffer_size, N_samples)
        idx     = np.random.choice(N_samples, size=n_store, replace=False)
        X_buf   = X_np[idx].astype(np.float32)
        y_buf   = y_np[idx]

        # ── empirical P(Y) ────────────────────────
        unique, counts = np.unique(y_np, return_counts=True)
        label_dist = pd.Series(
            counts / counts.sum(), index=unique
        )

        self.buffers[domain_key] = {
            'X'         : X_buf,
            'label_dist': label_dist,
        }

        print(
            f"  [Client {self.client_id}] "
            f"Buffer_{domain_key} stored ✓  "
            f"| {n_store} samples of {N_samples} "
            f"| input_dim={X_buf.shape[1]} "
            f"| classes={label_dist.to_dict()} "
            f"| total buffers: {len(self.buffers)}"
        )

        # ── KS test: buffer vs full training data ────────────────
        # Since the buffer IS a random sample of the real data,
        # the KS test should fail to reject H0 (p >> 0.05),
        # confirming the buffer faithfully represents the domain.
        from evaluate_generator import evaluate_generator
        self.buffer_eval_results[domain_key] = evaluate_generator(
            domain_key  = domain_key,
            real_X_np   = X_np,           # full training set
            generator   = {},             # not used when syn_X_np provided
            n_synthetic = n_store,
            save_plots  = True,
            plots_dir   = 'results/replay_plots',
            device      = str(self.device),
            syn_X_np    = X_buf,          # the buffer = real samples
        )

    # ═══════════════════════════════════════════════════════
    # PRIVATE — _build_replay_loader()
    # ═══════════════════════════════════════════════════════
    def _build_replay_loader(self, exclude_domain: str):
        """
        Build a DataLoader from the stored real-sample buffers
        of all past domains (excluding the current one).

        Returns None if no past domains have been seen yet
        (i.e., at the very first time step).
        """
        past_bufs = {
            k: v for k, v in self.buffers.items()
            if k != exclude_domain
        }

        if not past_bufs:
            return None

        all_X, all_y = [], []

        for domain_key, buf in past_bufs.items():
            X_buf = buf['X']               # already float32 real data
            n_buf = len(X_buf)

            # draw labels from empirical distribution
            y_buf = np.random.choice(
                buf['label_dist'].index,
                size = n_buf,
                p    = buf['label_dist'].values
            ).astype(np.int64)

            all_X.append(X_buf)
            all_y.append(y_buf)

            print(
                f"  [Client {self.client_id}] "
                f"← Replayed {n_buf} real "
                f"samples from Buffer_{domain_key}"
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
            f"{len(y_all)} real samples from "
            f"{len(past_bufs)} past domain(s) "
            f"| tensor shape: {tuple(X_tensor.shape)}"
        )
        return loader
