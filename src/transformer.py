"""
Temporal Transformer for water-system anomaly detection.

Architecture:
    Input (batch, seq_len, n_features)
        -> Linear projection  (n_features -> d_model)
        -> Sinusoidal positional encoding
        -> Transformer Encoder (N layers, multi-head self-attention)
        -> Mean pooling over time
        -> LayerNorm -> Dropout -> Linear(d_model -> 1)
        -> scalar logit  (use BCEWithLogitsLoss during training)
"""
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Chosen over learnable embeddings because:
    - sequence length is fixed (48 steps)
    - sinusoidal encoding generalises to unseen positions
    - no additional parameters to learn

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Register as buffer: moves with model.to(device), not a parameter
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer classifier for binary anomaly detection.

    Input:  (batch, seq_len, n_features)
    Output: (batch,)  — raw logits

    Pooling strategy: mean over time dimension.
    Chosen because it is stable, parameter-free, and captures
    the full sequence context rather than relying on a single position.
    """

    def __init__(self,
                 n_features:     int,
                 d_model:        int   = 64,
                 nhead:          int   = 4,
                 num_layers:     int   = 2,
                 dim_feedforward: int  = 128,
                 dropout:        float = 0.1,
                 max_seq_len:    int   = 512):
        super().__init__()

        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        # 1. Input projection: map raw features to model dimension
        self.input_proj = nn.Linear(n_features, d_model)

        # 2. Sinusoidal positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,   # (batch, seq, d_model) convention
            norm_first=False,   # post-norm (standard)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 4. Classification head
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch,)
        """
        # Project to d_model
        x = self.input_proj(x)              # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_enc(x)                 # (batch, seq_len, d_model)

        # Transformer encoder
        x = self.encoder(x)                 # (batch, seq_len, d_model)

        # Mean pool over time
        x = x.mean(dim=1)                   # (batch, d_model)

        # Classification head
        x = self.drop(x)
        logits = self.head(x).squeeze(-1)   # (batch,)
        return logits

    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Extract attention weights from each encoder layer.
        Returns list of (batch, nhead, seq_len, seq_len) tensors.
        Note: requires forward hooks — used only for analysis, not training.
        """
        weights = []
        hooks   = []

        def make_hook(layer_idx):
            def hook(module, inp, out):
                # TransformerEncoderLayer stores attn_output_weights
                # We re-run self-attention with need_weights=True
                pass
            return hook

        # Simpler approach: re-run each layer manually
        self.eval()
        with torch.no_grad():
            h = self.input_proj(x)
            h = self.pos_enc(h)
            for layer in self.encoder.layers:
                # Extract attention weights via the self-attention sub-module
                attn_out, attn_w = layer.self_attn(
                    h, h, h, need_weights=True, average_attn_weights=False
                )
                weights.append(attn_w.detach().cpu())
                # Continue through the rest of the layer normally
                h = layer(h)
        return weights

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
