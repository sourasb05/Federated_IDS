# models_pcflta.py
#
# Model components for PCFL-TA
# (Trajectory-Aligned Clustered Federated Learning).
#
# Architecture split:
#   Backbone (Φ) — shared LSTM; averaged globally every round
#   Head     (Ψ) — per-cluster FC classifier; averaged within cluster only
#
# The split lets the backbone learn universal traffic representations
# while each cluster head specialises for one attack-type family.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════
# 1. SHARED BACKBONE  (Φ)
# ═══════════════════════════════════════════════════════
class LSTMBackbone(nn.Module):
    """
    LSTM shared backbone — Φ in the PCFL-TA paper.

    Input : (B, 1, input_dim)  — same shape as utils.load_data output
    Output: feat (B, hidden_dim) — last-timestep hidden state used both
            for classification and for attack-signature extraction.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns last-timestep hidden state (B, hidden_dim)."""
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))   # (B, T, hidden_dim)
        return out[:, -1, :]               # last timestep


# ═══════════════════════════════════════════════════════
# 2. CLUSTER EXPERT HEAD  (Ψₖ)
# ═══════════════════════════════════════════════════════
class ClassifierHead(nn.Module):
    """
    Two-layer FC classifier — Ψₖ in the PCFL-TA paper.

    One head per cluster (k = 0 … n_clusters-1).
    Only clients assigned to cluster k contribute to Ψₖ's update.

    Input : feat (B, hidden_dim)
    Output: logits (B, output_dim)
    """

    def __init__(self, hidden_dim: int, fc_hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim,    fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(feat)))


# ═══════════════════════════════════════════════════════
# 3. COMBINED CLIENT MODEL  (Φ + Ψₖ)
# ═══════════════════════════════════════════════════════
class PCFLTALocalModel(nn.Module):
    """
    Full client-side model: backbone + one active expert head.

    The server reconstructs this for each client before training,
    loading the global backbone weights and the client's assigned
    cluster-head weights.  After training, backbone and head weights
    are extracted separately for aggregation.

    forward() returns:
        logits (B, output_dim)   — for cross-entropy loss
        feat   (B, hidden_dim)   — backbone representation used for
                                   attack-signature extraction (V⃗ᵢ)
    """

    def __init__(self, backbone: LSTMBackbone, head: ClassifierHead):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x: torch.Tensor):
        feat   = self.backbone(x)   # (B, hidden_dim)
        logits = self.head(feat)    # (B, n_classes)
        return logits, feat
