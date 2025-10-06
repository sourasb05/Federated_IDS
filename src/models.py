import torch.nn as nn
import torch.nn.functional as F
import torch

class LSTMClassifier(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, num_layers=1, fc_hidden_dim=64, head_dropout: float = 0.0):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))   # out: (B, T, H)

        # Take last time step
        feat = out[:, -1, :]                        # (B, H)

        # Two-layer head
        feat = F.relu(self.fc1(feat))
        feat = self.dropout(feat)
        logits = self.fc2(feat)                     # raw logits (B, C)

        return logits, (h_n, c_n)
        

class LSTMModelWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0, bidirectional=False):
        super(LSTMModelWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.attn_fc = nn.Linear(hidden_size * self.num_directions, 1)
        self.output_fc = nn.Linear(hidden_size * self.num_directions, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def attention_net(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim*num_directions]
        attn_weights = self.attn_fc(lstm_output)  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # [batch_size, seq_len, 1]
        weighted_output = lstm_output * attn_weights  # [batch_size, seq_len, hidden_dim*num_directions]
        context_vector = weighted_output.sum(dim=1)  # [batch_size, hidden_dim*num_directions]
        return context_vector

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)

        lstm_out, _ = self.lstm(x, (h0, c0))  # [batch_size, seq_len, hidden_dim*num_directions]

        # Apply attention mechanism
        context = self.attention_net(lstm_out)

        # Pass through output layer
        output = self.output_fc(context)
        return output, context

