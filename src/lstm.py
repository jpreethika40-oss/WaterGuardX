"""LSTM classifier for water-system anomaly detection."""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for binary anomaly classification.

    Input:  (batch, seq_len, input_size)
    Output: (batch,)  — raw logits (use BCEWithLogitsLoss during training)

    Architecture:
        Input sequence
            -> LSTM (stacked, optionally bidirectional)
            -> last hidden state (or mean-pool over time)
            -> LayerNorm
            -> Dropout
            -> Linear -> scalar logit
    """

    def __init__(self,
                 input_size:   int,
                 hidden_size:  int  = 128,
                 num_layers:   int  = 2,
                 dropout:      float = 0.3,
                 bidirectional: bool = True):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.directions    = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * self.directions
        self.norm    = nn.LayerNorm(lstm_out_dim)
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Linear(lstm_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch,)
        """
        # out: (batch, seq_len, hidden*directions)
        out, _ = self.lstm(x)

        # Mean-pool over the time dimension — more stable than last-step only
        # and captures the full sequence context
        pooled = out.mean(dim=1)          # (batch, hidden*directions)

        pooled = self.norm(pooled)
        pooled = self.drop(pooled)
        logits = self.head(pooled).squeeze(-1)   # (batch,)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability for class 1."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
