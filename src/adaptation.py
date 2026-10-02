"""
Phase 8 — Adaptive Learning Under Distribution Shift.

Provides modular functions for:
1. Chronological splitting of shifted incoming data (Adaptation pool vs. Evaluation pool).
2. Constructing replay buffer datasets (mixing adaptation stream with training reference data).
3. Controlled fine-tuning of the SSL Temporal Transformer.
4. Model evaluation on normal and shifted distributions.
5. Adaptation safety checks and acceptance criteria to prevent catastrophic forgetting.
6. Checkpoint management.
"""
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import (DATA_PROC, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SEED,
                    SENSOR_COLS, SEQUENCE_LENGTH, STRIDE, TARGET_COL,
                    TIMESTAMP_COL)
from datasets import WaterSensorDataset
from evaluation import compute_metrics, select_threshold_f1
from self_supervised import SSLTransformer
from training import set_seed


class ReplayDataset(Dataset):
    """
    Combines shifted adaptation sequences with a replay buffer of reference
    training sequences to prevent catastrophic forgetting.
    """
    def __init__(self, adapt_windows: np.ndarray, adapt_labels: np.ndarray,
                 replay_windows: Optional[np.ndarray] = None,
                 replay_labels: Optional[np.ndarray] = None):
        if replay_windows is not None and len(replay_windows) > 0:
            self.windows = np.concatenate([adapt_windows, replay_windows], axis=0)
            self.labels = np.concatenate([adapt_labels, replay_labels], axis=0)
            # Shuffle combined dataset
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(len(self.labels))
            self.windows = self.windows[perm]
            self.labels = self.labels[perm]
        else:
            self.windows = adapt_windows
            self.labels = adapt_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.windows[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))


def split_chronological_adaptation(
    shifted_df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    split_date: str = '2018-12-01',
    val_ratio: float = 0.30
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split incoming shifted data chronologically into:
      1. adapt_train_df : training subset for model adaptation (first (1-val_ratio) of pre-split)
      2. adapt_val_df   : validation subset for safety check / threshold tuning
      3. eval_df        : untouched final evaluation subset (post-split date)

    Guarantees ZERO data leakage between adaptation and final test evaluation.
    """
    df = shifted_df.copy()
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        adapt_pool = df[df[timestamp_col] < split_date].reset_index(drop=True)
        eval_df = df[df[timestamp_col] >= split_date].reset_index(drop=True)
    else:
        n_total = len(df)
        n_half = n_total // 2
        adapt_pool = df.iloc[:n_half].reset_index(drop=True)
        eval_df = df.iloc[n_half:].reset_index(drop=True)

    n_adapt = len(adapt_pool)
    n_train = int(n_adapt * (1.0 - val_ratio))
    adapt_train_df = adapt_pool.iloc[:n_train].reset_index(drop=True)
    adapt_val_df = adapt_pool.iloc[n_train:].reset_index(drop=True)

    return adapt_train_df, adapt_val_df, eval_df


def build_replay_dataset(
    adapt_df: pd.DataFrame,
    train_df: pd.DataFrame,
    feat_cols: List[str],
    replay_ratio: float = 0.20,
    seq_len: int = SEQUENCE_LENGTH,
    stride: int = STRIDE,
    seed: int = SEED,
) -> ReplayDataset:
    """
    Extract sliding windows from adaptation data and sample replay sequences
    from original reference training data.
    """
    adapt_ds = WaterSensorDataset(adapt_df, feat_cols, seq_len=seq_len, stride=stride)
    adapt_win = adapt_ds.windows
    adapt_lbl = adapt_ds.labels

    if replay_ratio > 0.0 and len(train_df) > 0:
        train_ds = WaterSensorDataset(train_df, feat_cols, seq_len=seq_len, stride=stride)
        n_replay = int(len(adapt_win) * (replay_ratio / (1.0 - replay_ratio)))
        n_replay = min(n_replay, len(train_ds))

        rng = np.random.default_rng(seed)
        replay_indices = rng.choice(len(train_ds), size=n_replay, replace=False)
        replay_win = train_ds.windows[replay_indices]
        replay_lbl = train_ds.labels[replay_indices]
    else:
        replay_win, replay_lbl = None, None

    return ReplayDataset(adapt_win, adapt_lbl, replay_win, replay_lbl)


def load_ssl_model(
    checkpoint_path: Path,
    config_path: Path,
    device: str = 'cpu'
) -> Tuple[SSLTransformer, dict]:
    """Load pretrained SSLTransformer from checkpoint and configuration."""
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    model = SSLTransformer(
        n_features=cfg['n_features'],
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_layers=cfg['num_layers'],
        dim_feedforward=cfg['dim_feedforward'],
        dropout=cfg['dropout'],
        max_seq_len=SEQUENCE_LENGTH + 10,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def evaluate_model_on_data(
    model: nn.Module,
    df_data: pd.DataFrame,
    feat_cols: List[str],
    threshold: float,
    batch_size: int = 256,
    device: str = 'cpu',
    stride: int = STRIDE,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Run model on a DataFrame split and compute metrics."""
    ds = WaterSensorDataset(df_data, feat_cols, label_strategy='last', stride=stride)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            logits = model.forward_classify(X_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_b.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds = (probs >= threshold).astype(int)
    metrics = compute_metrics(labels, preds, probs)
    return metrics, probs, labels


def run_controlled_adaptation(
    model: SSLTransformer,
    train_dataset: Dataset,
    val_df: pd.DataFrame,
    feat_cols: List[str],
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    device: str = 'cpu',
) -> Tuple[SSLTransformer, float, list, list]:
    """
    Perform controlled fine-tuning of SSL Transformer on adaptation data.
    Returns:
      adapted_model, new_threshold, train_losses, val_f1s
    """
    adapted_model = copy.deepcopy(model)
    adapted_model.train()

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(adapted_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_f1s = []
    best_val_f1 = -1.0
    best_threshold = 0.5
    best_state = None

    for epoch in range(1, epochs + 1):
        adapted_model.train()
        total_loss = 0.0
        n_batches = 0

        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = adapted_model.forward_classify(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Validate on adaptation validation split to tune threshold & track F1
        val_met, val_probs, val_labels = evaluate_model_on_data(
            adapted_model, val_df, feat_cols, threshold=0.5, batch_size=batch_size, device=device
        )
        epoch_thresh = select_threshold_f1(val_labels, val_probs)
        pred_val = (val_probs >= epoch_thresh).astype(int)
        epoch_f1 = compute_metrics(val_labels, pred_val, val_probs)['f1']
        val_f1s.append(epoch_f1)

        if epoch_f1 > best_val_f1:
            best_val_f1 = epoch_f1
            best_threshold = epoch_thresh
            best_state = {k: v.cpu().clone() for k, v in adapted_model.state_dict().items()}

    if best_state is not None:
        adapted_model.load_state_dict(best_state)

    adapted_model.eval()
    return adapted_model, best_threshold, train_losses, val_f1s


def check_acceptance_criteria(
    orig_f1_shift_val: float,
    cand_f1_shift_val: float,
    orig_f1_orig_val: float,
    cand_f1_orig_val: float,
    max_orig_degradation: float = 0.05
) -> Tuple[bool, dict]:
    """
    Acceptance Criteria for Promoting Candidate Adapted Model:
      1. Must improve on shifted validation data:
         cand_f1_shift_val > orig_f1_shift_val
      2. Must not suffer unacceptable degradation on original validation data:
         cand_f1_orig_val >= orig_f1_orig_val - max_orig_degradation
    """
    delta_shift = cand_f1_shift_val - orig_f1_shift_val
    delta_orig = cand_f1_orig_val - orig_f1_orig_val

    rule1_pass = bool(delta_shift > 0.0)
    rule2_pass = bool(delta_orig >= -max_orig_degradation)
    accepted = bool(rule1_pass and rule2_pass)

    details = {
        'orig_f1_shift_val': round(orig_f1_shift_val, 4),
        'cand_f1_shift_val': round(cand_f1_shift_val, 4),
        'delta_shift_val':   round(delta_shift, 4),
        'rule1_pass':        rule1_pass,
        'orig_f1_orig_val':  round(orig_f1_orig_val, 4),
        'cand_f1_orig_val':  round(cand_f1_orig_val, 4),
        'delta_orig_val':    round(delta_orig, 4),
        'max_orig_degradation_allowed': max_orig_degradation,
        'rule2_pass':        rule2_pass,
        'accepted':          accepted,
    }
    return accepted, details
