"""
Reusable PyTorch training loop.
Used by LSTM (Phase 4) and Transformer (Phase 5+).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from config import SEED


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """
    Stop training when validation F1 stops improving.
    Saves the best model weights in memory.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -np.inf
        self.counter    = 0
        self.best_state = None
        self.best_epoch = 0

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Returns True if training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        """Load the best saved weights back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def run_epoch(model: nn.Module,
              loader: DataLoader,
              criterion: nn.Module,
              optimizer=None,
              device: str = 'cpu',
              threshold: float = 0.5) -> dict:
    """
    Run one training or validation epoch.

    Args:
        model:     PyTorch model
        loader:    DataLoader
        criterion: loss function
        optimizer: if None, runs in eval mode (no gradient)
        device:    'cpu' or 'cuda'
        threshold: classification threshold for F1 calculation

    Returns:
        dict with loss, f1, all probabilities and labels
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_probs  = []
    all_labels = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_batch.detach().cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels).astype(int)
    avg_loss   = total_loss / len(all_labels)
    preds      = (all_probs >= threshold).astype(int)
    f1         = f1_score(all_labels, preds, zero_division=0)

    return {
        'loss':   avg_loss,
        'f1':     f1,
        'probs':  all_probs,
        'labels': all_labels,
    }


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                criterion:    nn.Module,
                optimizer,
                scheduler,
                epochs:    int = 50,
                patience:  int = 10,
                device:    str = 'cpu',
                verbose:   bool = True) -> dict:
    """
    Full training loop with early stopping and LR scheduling.

    Returns:
        history dict with per-epoch train/val loss and F1,
        plus best_epoch and best_val_f1.
    """
    set_seed()
    es = EarlyStopping(patience=patience)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1':   [], 'val_f1':   [],
        'lr':         [],
    }

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device)
        vl = run_epoch(model, val_loader,   criterion, None,      device)

        history['train_loss'].append(tr['loss'])
        history['val_loss'].append(vl['loss'])
        history['train_f1'].append(tr['f1'])
        history['val_f1'].append(vl['f1'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(vl['loss'])
        else:
            scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d} | "
                  f"train_loss={tr['loss']:.4f} val_loss={vl['loss']:.4f} | "
                  f"train_f1={tr['f1']:.4f} val_f1={vl['f1']:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if es.step(vl['f1'], model, epoch):
            if verbose:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best epoch={es.best_epoch}, "
                      f"best val_f1={es.best_score:.4f})")
            break

    es.restore(model)
    history['best_epoch']    = es.best_epoch
    history['best_val_f1']   = es.best_score
    history['best_val_loss'] = history['val_loss'][es.best_epoch - 1]
    return history
