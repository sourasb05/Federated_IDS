# client_PCFLTA.py
#
# Federated client for PCFL-TA
# (Trajectory-Aligned Clustered Federated Learning).
#
# Each round the client:
#   1. Receives: global backbone Φₜ + assigned cluster head Ψ_cluster
#   2. Runs local training on current-domain data (lines 4 of Algorithm 1)
#   3. Extracts hidden states for normal vs attack samples (line 5)
#   4. Computes attack signature V⃗ᵢ = mean(H_atk) - mean(H_norm) (line 6)
#   5. Returns: backbone weights, head weights, V⃗ᵢ  (line 7)
#
# No generative replay — the server clusters clients by their signatures
# and ensures each cluster head specialises for one attack family.

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import utils
from models_pcflta import LSTMBackbone, ClassifierHead, PCFLTALocalModel


class ClientPCFLTA:

    def __init__(self, client_id, args, domain_path,
                 assigned_domains, device,
                 hidden_dim: int, fc_hidden_dim: int):
        """
        Args:
            hidden_dim    : LSTM hidden size (same as backbone)
            fc_hidden_dim : hidden size of classifier head FC layer
        """
        self.client_id        = client_id
        self.args             = args
        self.domain_path      = domain_path
        self.assigned_domains = assigned_domains
        self.device           = device
        self.hidden_dim       = hidden_dim
        self.fc_hidden_dim    = fc_hidden_dim

        # ── data loaders ──────────────────────────
        self.train_domains_loader = {}
        self.test_domains_loader  = {}

        domains = utils.create_domains(domain_path, assigned_domains)
        for key, files in domains.items():
            self.train_domains_loader[key], \
            self.test_domains_loader[key] = utils.load_data(
                self.domain_path,
                key,
                files,
                window_size = self.args.window_size,
                step_size   = self.args.step_size,
                batch_size  = self.args.batch_size
            )

        self.domain_keys = list(self.train_domains_loader.keys())

        # ── local model (re-built each round from server weights) ──
        self.input_dim  = args.input_size
        self.output_dim = args.output_size

        # Placeholders — populated in train()
        self.local_model: PCFLTALocalModel = None
        self.optimizer  = None
        self.criterion  = nn.CrossEntropyLoss()

    # ═══════════════════════════════════════════════════════
    # PUBLIC — train()
    # ═══════════════════════════════════════════════════════
    def train(self, backbone_state: dict, head_state: dict,
              cluster_id: int, time_step: int):
        """
        Local update: Algorithm 1, lines 4-7.

        Args:
            backbone_state : global backbone weights (Φₜ)
            head_state     : assigned cluster head weights (Ψ_cluster)
            cluster_id     : index of the client's current cluster
            time_step      : domain index to train on

        Returns:
            backbone_weights : dict  — trained backbone state dict
            head_weights     : dict  — trained head state dict
            signature        : np.ndarray (hidden_dim,) — V⃗ᵢ
        """
        current_domain = self.domain_keys[time_step]

        # ── build model from server-provided weights ────
        backbone = LSTMBackbone(
            self.input_dim, self.hidden_dim, self.args.num_layers
        ).to(self.device)
        head = ClassifierHead(
            self.hidden_dim, self.fc_hidden_dim, self.output_dim
        ).to(self.device)

        backbone.load_state_dict(copy.deepcopy(backbone_state))
        head.load_state_dict(copy.deepcopy(head_state))

        self.local_model = PCFLTALocalModel(backbone, head).to(self.device)
        self.optimizer   = optim.Adam(
            self.local_model.parameters(), lr=self.args.lr
        )

        # ── local training ──────────────────────────────
        self.local_model.train()
        for epoch in range(self.args.local_epochs):
            for data, target in self.train_domains_loader[current_domain]:
                data   = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.local_model(data)
                loss      = self.criterion(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.local_model.parameters(), max_norm=5.0
                )
                self.optimizer.step()

        # ── extract attack signature V⃗ᵢ ────────────────
        # Line 5-6: H_norm, H_atk ← ExtractHiddenStates(f_φ)
        #           V⃗ᵢ ← mean(H_atk) − mean(H_norm)
        signature = self._extract_signature(current_domain)

        print(
            f"  [Client {self.client_id}] "
            f"cluster={cluster_id}  domain={current_domain}  "
            f"sig_norm={np.linalg.norm(signature):.4f}"
        )

        return (
            copy.deepcopy(self.local_model.backbone.state_dict()),
            copy.deepcopy(self.local_model.head.state_dict()),
            signature,
        )

    # ═══════════════════════════════════════════════════════
    # PUBLIC — evaluate_global_model()
    # ═══════════════════════════════════════════════════════
    def evaluate_global_model(self, backbone_state: dict,
                               head_state: dict, time_step: int):
        """
        Evaluate the global backbone + assigned cluster head on the
        current-domain test set.
        """
        backbone = LSTMBackbone(
            self.input_dim, self.hidden_dim, self.args.num_layers
        ).to(self.device)
        head = ClassifierHead(
            self.hidden_dim, self.fc_hidden_dim, self.output_dim
        ).to(self.device)
        backbone.load_state_dict(backbone_state)
        head.load_state_dict(head_state)

        eval_model = PCFLTALocalModel(backbone, head).to(self.device)
        eval_model.eval()

        current_domain = self.domain_keys[time_step]
        total_loss     = 0.0
        all_preds      = []
        all_targets    = []

        with torch.no_grad():
            for data, target in self.test_domains_loader[current_domain]:
                data   = data.to(self.device)
                target = target.to(self.device)

                logits, _ = eval_model(data)
                total_loss += self.criterion(logits, target).item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        n_batches = len(self.test_domains_loader[current_domain])
        eval_loss = total_loss / max(n_batches, 1)
        accuracy  = accuracy_score(all_targets, all_preds)
        f1        = f1_score(
            all_targets, all_preds,
            average='macro', zero_division=0
        )

        print(
            f"  [Client {self.client_id}] "
            f"domain={current_domain}  "
            f"Loss={eval_loss:.4f}  Acc={accuracy:.4f}  F1={f1:.4f}"
        )
        return eval_loss, accuracy, f1

    # ═══════════════════════════════════════════════════════
    # PRIVATE — _extract_signature()
    # ═══════════════════════════════════════════════════════
    def _extract_signature(self, domain_key: str) -> np.ndarray:
        """
        Algorithm 1, lines 5-6:
            H_norm, H_atk ← ExtractHiddenStates(f_φ)
            V⃗ᵢ ← mean(H_atk) − mean(H_norm)

        The backbone (f_φ) maps each input window to a hidden
        representation.  Separating representations by class and
        taking their difference creates an 'attack trajectory' vector
        that encodes how the model distinguishes attack from normal
        traffic — its direction captures the attack type family.
        """
        self.local_model.eval()

        h_norm_list: list = []
        h_atk_list:  list = []

        with torch.no_grad():
            for data, target in self.train_domains_loader[domain_key]:
                data   = data.to(self.device)
                target = target.to(self.device)

                # forward through backbone only
                feat = self.local_model.backbone(data)  # (B, hidden_dim)

                feat_np   = feat.cpu().numpy()
                target_np = target.cpu().numpy()

                norm_mask = (target_np == 0)
                atk_mask  = (target_np == 1)

                if norm_mask.any():
                    h_norm_list.append(feat_np[norm_mask])
                if atk_mask.any():
                    h_atk_list.append(feat_np[atk_mask])

        dim = self.hidden_dim

        if h_atk_list:
            mean_atk = np.vstack(h_atk_list).mean(axis=0)
        else:
            mean_atk = np.zeros(dim, dtype=np.float32)

        if h_norm_list:
            mean_norm = np.vstack(h_norm_list).mean(axis=0)
        else:
            mean_norm = np.zeros(dim, dtype=np.float32)

        return (mean_atk - mean_norm).astype(np.float32)
