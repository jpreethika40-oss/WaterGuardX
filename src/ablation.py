"""
Phase 9 — Ablation Study and Component Contribution Analysis.

Provides modular functions for:
1. Loading and adapting both Baseline and SSL Temporal Transformers.
2. Standardized generic evaluation across all ablation configurations.
3. Component contribution calculations (SSL gain, Adaptation gain, System synergy).
4. Error analysis and computational cost profiling across configurations.
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
from transformer import TemporalTransformer


def load_baseline_transformer(
    checkpoint_path: Path,
    config_path: Path,
    device: str = 'cpu'
) -> Tuple[TemporalTransformer, dict]:
    """Load pretrained baseline TemporalTransformer (without SSL)."""
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    model = TemporalTransformer(
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


def evaluate_model_generic(
    model: nn.Module,
    df_data: pd.DataFrame,
    feat_cols: List[str],
    threshold: float,
    batch_size: int = 256,
    device: str = 'cpu',
    stride: int = STRIDE,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Run any model (TemporalTransformer or SSLTransformer) on a DataFrame split.
    Returns metrics dict, probabilities, and ground-truth labels.
    """
    ds = WaterSensorDataset(df_data, feat_cols, label_strategy='last', stride=stride)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            if hasattr(model, 'forward_classify'):
                logits = model.forward_classify(X_b)
            else:
                logits = model(X_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_b.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds = (probs >= threshold).astype(int)
    metrics = compute_metrics(labels, preds, probs)
    return metrics, probs, labels


def adapt_generic_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_df: pd.DataFrame,
    feat_cols: List[str],
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    device: str = 'cpu',
) -> Tuple[nn.Module, float, list, list, float]:
    """
    Fine-tune any Temporal Transformer architecture on adaptation data with replay.
    Returns: adapted_model, calibrated_threshold, train_losses, val_f1s, duration_seconds.
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

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        adapted_model.train()
        total_loss = 0.0
        n_batches = 0

        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            if hasattr(adapted_model, 'forward_classify'):
                logits = adapted_model.forward_classify(X_b)
            else:
                logits = adapted_model(X_b)

            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Validate on adaptation validation split to calibrate threshold
        val_met, val_probs, val_labels = evaluate_model_generic(
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

    t1 = time.perf_counter()
    duration = round(t1 - t0, 2)

    if best_state is not None:
        adapted_model.load_state_dict(best_state)

    adapted_model.eval()
    return adapted_model, best_threshold, train_losses, val_f1s, duration


def compute_component_contributions(
    metrics_a: dict, # Baseline (No SSL, No Adapt)
    metrics_b: dict, # SSL (SSL, No Adapt)
    metrics_c: dict, # Base + Adapt (No SSL, Adapt)
    metrics_d: dict, # SSL + Adapt (SSL, Adapt)
    metrics_e: dict, # Full System
) -> pd.DataFrame:
    """
    Calculate component contributions and interactions across ablation configurations.
    """
    records = [
        {
            'Component / Effect': 'SSL Contribution on Normal Data',
            'Comparison': 'Config B vs. Config A (Normal)',
            'Metric': 'F1-Score',
            'Base Value': round(metrics_a['normal']['f1'], 4),
            'With Component': round(metrics_b['normal']['f1'], 4),
            'Delta': round(metrics_b['normal']['f1'] - metrics_a['normal']['f1'], 4),
        },
        {
            'Component / Effect': 'SSL Contribution on Shifted Data (Pre-Adaptation)',
            'Comparison': 'Config B vs. Config A (Shifted)',
            'Metric': 'F1-Score',
            'Base Value': round(metrics_a['shifted']['f1'], 4),
            'With Component': round(metrics_b['shifted']['f1'], 4),
            'Delta': round(metrics_b['shifted']['f1'] - metrics_a['shifted']['f1'], 4),
        },
        {
            'Component / Effect': 'SSL Representation Advantage (Shifted ROC-AUC)',
            'Comparison': 'Config B vs. Config A (Shifted)',
            'Metric': 'ROC-AUC',
            'Base Value': round(metrics_a['shifted']['roc_auc'], 4),
            'With Component': round(metrics_b['shifted']['roc_auc'], 4),
            'Delta': round(metrics_b['shifted']['roc_auc'] - metrics_a['shifted']['roc_auc'], 4),
        },
        {
            'Component / Effect': 'Adaptation Contribution without SSL',
            'Comparison': 'Config C vs. Config A (Shifted Eval)',
            'Metric': 'F1-Score',
            'Base Value': round(metrics_a['shifted']['f1'], 4),
            'With Component': round(metrics_c['adapted']['f1'], 4),
            'Delta': round(metrics_c['adapted']['f1'] - metrics_a['shifted']['f1'], 4),
        },
        {
            'Component / Effect': 'Adaptation Contribution with SSL',
            'Comparison': 'Config D vs. Config B (Shifted Eval)',
            'Metric': 'F1-Score',
            'Base Value': round(metrics_b['shifted']['f1'], 4),
            'With Component': round(metrics_d['adapted']['f1'], 4),
            'Delta': round(metrics_d['adapted']['f1'] - metrics_b['shifted']['f1'], 4),
        },
        {
            'Component / Effect': 'SSL Advantage under Adapted Regime',
            'Comparison': 'Config D vs. Config C (Adapted Shifted)',
            'Metric': 'F1-Score',
            'Base Value': round(metrics_c['adapted']['f1'], 4),
            'With Component': round(metrics_d['adapted']['f1'], 4),
            'Delta': round(metrics_d['adapted']['f1'] - metrics_c['adapted']['f1'], 4),
        },
        {
            'Component / Effect': 'Full System Synergistic Gain',
            'Comparison': 'Config E (Full) vs. Config A (Baseline)',
            'Metric': 'Shifted F1-Score',
            'Base Value': round(metrics_a['shifted']['f1'], 4),
            'With Component': round(metrics_e['adapted']['f1'], 4),
            'Delta': round(metrics_e['adapted']['f1'] - metrics_a['shifted']['f1'], 4),
        },
        {
            'Component / Effect': 'Full System Normal Retention',
            'Comparison': 'Config E (Full) vs. Config A (Baseline)',
            'Metric': 'Normal F1-Score',
            'Base Value': round(metrics_a['normal']['f1'], 4),
            'With Component': round(metrics_e['normal']['f1'], 4),
            'Delta': round(metrics_e['normal']['f1'] - metrics_a['normal']['f1'], 4),
        },
    ]
    return pd.DataFrame(records)
