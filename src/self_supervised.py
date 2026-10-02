"""
Phase 6 — Self-Supervised Masked Time-Series Reconstruction.

Architecture:
    SSLTransformer
        input_proj  → positional encoding → TransformerEncoder
        ↓ (encoder shared with classifier)
        ReconstructionHead  → (batch, seq_len, n_features)   [pretraining]
        ClassificationHead  → (batch,)  logit                [fine-tuning]

Pretraining loss: masked MSE — computed only on artificially masked positions.
Fine-tuning loss: BCEWithLogitsLoss — same as Phase 5.
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import SEED


# ── Positional encoding (reuse same design as Phase 5) ───────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ── SSL Transformer ───────────────────────────────────────────────────────────

class SSLTransformer(nn.Module):
    """
    Transformer with two heads:
      - reconstruction_head: used during SSL pretraining
      - classification_head: used during supervised fine-tuning

    The encoder (input_proj + pos_enc + transformer_encoder) is shared.
    """

    def __init__(self,
                 n_features:      int,
                 d_model:         int   = 32,
                 nhead:           int   = 2,
                 num_layers:      int   = 2,
                 dim_feedforward: int   = 64,
                 dropout:         float = 0.1,
                 max_seq_len:     int   = 512):
        super().__init__()
        assert d_model % nhead == 0

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = SinusoidalPE(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu',
            batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Reconstruction head: project back to input space at every timestep
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_features),
        )

        # Classification head: mean-pool → dropout → linear
        self.drop = nn.Dropout(dropout)
        self.classification_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: (batch, seq_len, n_features) → (batch, seq_len, d_model)"""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        return self.encoder(x)

    def forward_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """SSL pretraining forward: returns (batch, seq_len, n_features)"""
        h = self.encode(x)
        return self.reconstruction_head(h)

    def forward_classify(self, x: torch.Tensor) -> torch.Tensor:
        """Supervised fine-tuning forward: returns (batch,) logits"""
        h = self.encode(x)
        pooled = h.mean(dim=1)
        return self.classification_head(self.drop(pooled)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward executes classification head."""
        return self.forward_classify(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Masked MSE loss ───────────────────────────────────────────────────────────

def masked_mse_loss(pred: torch.Tensor,
                    target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """
    MSE computed only on masked positions.

    Args:
        pred:   (batch, seq_len, n_features) — model reconstruction
        target: (batch, seq_len, n_features) — original (unmasked) values
        mask:   (batch, seq_len, n_features) — 1 where artificially masked

    Returns:
        scalar loss
    """
    diff = (pred - target) ** 2
    n_masked = mask.sum()
    if n_masked == 0:
        return diff.mean()
    return (diff * mask).sum() / n_masked


# ── Pretraining loop ──────────────────────────────────────────────────────────

def pretrain_epoch(model: SSLTransformer,
                   loader: DataLoader,
                   optimizer,
                   device: str,
                   training: bool = True) -> float:
    """Run one SSL pretraining epoch. Returns mean masked MSE."""
    model.train() if training else model.eval()
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for masked_x, original_x, mask in loader:
            masked_x   = masked_x.to(device)
            original_x = original_x.to(device)
            mask       = mask.to(device)

            pred = model.forward_reconstruct(masked_x)
            loss = masked_mse_loss(pred, original_x, mask)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def pretrain_ssl(model: SSLTransformer,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 epochs:       int   = 10,
                 lr:           float = 1e-3,
                 weight_decay: float = 1e-4,
                 patience:     int   = 5,
                 device:       str   = 'cpu',
                 verbose:      bool  = True) -> dict:
    """
    Full SSL pretraining loop with early stopping on val reconstruction loss.

    Returns history dict.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    best_val_loss  = float('inf')
    best_state     = None
    best_epoch     = 0
    no_improve     = 0
    history        = {'train_loss': [], 'val_loss': [], 'epoch_time': []}
    total_start    = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss = pretrain_epoch(model, train_loader, optimizer, device, training=True)
        vl_loss = pretrain_epoch(model, val_loader,   optimizer, device, training=False)
        ep_time = time.perf_counter() - t0

        scheduler.step(vl_loss)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['epoch_time'].append(round(ep_time, 2))

        if verbose:
            print(f"  SSL Epoch {epoch:3d} | "
                  f"train_loss={tr_loss:.6f}  val_loss={vl_loss:.6f} | "
                  f"time={ep_time:.1f}s")

        if vl_loss < best_val_loss - 1e-6:
            best_val_loss = vl_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best epoch={best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history['best_epoch']     = best_epoch
    history['best_val_loss']  = best_val_loss
    history['total_time_s']   = round(time.perf_counter() - total_start, 2)
    return history


# ── Fine-tuning loop ──────────────────────────────────────────────────────────

def finetune_epoch(model: SSLTransformer,
                   loader: DataLoader,
                   criterion: nn.Module,
                   optimizer,
                   device: str,
                   training: bool = True) -> dict:
    """One supervised fine-tuning epoch. Returns loss, probs, labels."""
    model.train() if training else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()

            logits = model.forward_classify(X_batch)
            loss   = criterion(logits, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(y_batch.detach().cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels).astype(int)
    return {
        'loss':   total_loss / len(all_labels),
        'probs':  all_probs,
        'labels': all_labels,
    }
