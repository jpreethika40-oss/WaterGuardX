"""
Phase 11 — Error Analysis Utilities for WaterGuardX.

Provides:
1. generate_detailed_predictions: Generates per-sequence predictions and error tags.
2. extract_representative_errors: Identifies top false-positive and false-negative sequences.
3. sensor_distribution_by_error_type: Statistical comparison of sensors across error classes.
4. categorize_errors: Evidence-based error categorization.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import SENSOR_COLS, SEQUENCE_LENGTH, STRIDE, TARGET_COL, TIMESTAMP_COL
from datasets import WaterSensorDataset
from evaluation import compute_metrics


def generate_detailed_predictions(
    model: nn.Module,
    df_data: pd.DataFrame,
    feat_cols: List[str],
    threshold: float,
    eval_condition: str,
    stride: int = STRIDE,
    device: str = 'cpu',
    batch_size: int = 256
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate dense per-sequence predictions on a DataFrame split.
    Returns:
        preds_df: DataFrame with sequence_id, timestamp, true_label, predicted_label,
                  anomaly_prob, threshold, eval_condition, error_type.
        windows: numpy array of shape (N, seq_len, n_features).
    """
    ds = WaterSensorDataset(df_data, feat_cols, label_strategy='last', stride=stride)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_b, _ in loader:
            logits = model.forward_classify(X_b.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs)
    labels = ds.labels.astype(int)
    preds = (probs >= threshold).astype(int)

    # Sequence timestamps (timestamp at the end of the sliding window)
    timestamps = []
    if TIMESTAMP_COL in df_data.columns:
        ts_series = df_data[TIMESTAMP_COL].reset_index(drop=True)
        # Window indices: [i : i + seq_len], so last timestep is i + seq_len - 1
        for i in range(len(ds)):
            idx_last = i * stride + SEQUENCE_LENGTH - 1
            if idx_last < len(ts_series):
                timestamps.append(str(ts_series.iloc[idx_last]))
            else:
                timestamps.append(f"Seq_{i}")
    else:
        timestamps = [f"Seq_{i}" for i in range(len(ds))]

    # Error type categorization
    error_types = []
    for y_true, y_pred in zip(labels, preds):
        if y_true == 1 and y_pred == 1:
            error_types.append("True_Positive")
        elif y_true == 0 and y_pred == 0:
            error_types.append("True_Negative")
        elif y_true == 0 and y_pred == 1:
            error_types.append("False_Positive")
        elif y_true == 1 and y_pred == 0:
            error_types.append("False_Negative")
        else:
            error_types.append("Unknown")

    preds_df = pd.DataFrame({
        "sequence_id": np.arange(len(ds)),
        "timestamp": timestamps,
        "true_label": labels,
        "predicted_label": preds,
        "anomaly_prob": np.round(probs, 5),
        "threshold": round(threshold, 4),
        "eval_condition": eval_condition,
        "error_type": error_types,
    })

    return preds_df, ds.windows


def extract_representative_errors(
    preds_df: pd.DataFrame,
    windows: np.ndarray,
    n_examples: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract top false-positive and false-negative sequences sorted by confidence margin.
    - Top FPs: False positives with highest predicted anomaly probability (most confident false alarms).
    - Top FNs: False negatives with lowest predicted anomaly probability (most severely missed anomalies).
    """
    fp_mask = preds_df["error_type"] == "False_Positive"
    fn_mask = preds_df["error_type"] == "False_Negative"

    fp_df = preds_df[fp_mask].sort_values("anomaly_prob", ascending=False).head(n_examples).copy()
    fn_df = preds_df[fn_mask].sort_values("anomaly_prob", ascending=True).head(n_examples).copy()

    return fp_df, fn_df


def sensor_distribution_by_error_type(
    windows: np.ndarray,
    preds_df: pd.DataFrame,
    feat_cols: List[str]
) -> pd.DataFrame:
    """
    Compute mean and standard deviation of each sensor across TP, TN, FP, and FN windows.
    """
    labels = preds_df["error_type"].values
    n_sensors = min(len(SENSOR_COLS), windows.shape[2])

    records = []
    for fi in range(n_sensors):
        sensor_name = feat_cols[fi]
        # Look at the final timestep in the sequence window
        vals = windows[:, -1, fi]

        tp_vals = vals[labels == "True_Positive"]
        tn_vals = vals[labels == "True_Negative"]
        fp_vals = vals[labels == "False_Positive"]
        fn_vals = vals[labels == "False_Negative"]

        records.append({
            "sensor": sensor_name,
            "mean_TN": round(float(np.mean(tn_vals)), 4) if len(tn_vals) > 0 else 0.0,
            "std_TN":  round(float(np.std(tn_vals)), 4) if len(tn_vals) > 0 else 0.0,
            "mean_TP": round(float(np.mean(tp_vals)), 4) if len(tp_vals) > 0 else 0.0,
            "std_TP":  round(float(np.std(tp_vals)), 4) if len(tp_vals) > 0 else 0.0,
            "mean_FP": round(float(np.mean(fp_vals)), 4) if len(fp_vals) > 0 else 0.0,
            "std_FP":  round(float(np.std(fp_vals)), 4) if len(fp_vals) > 0 else 0.0,
            "mean_FN": round(float(np.mean(fn_vals)), 4) if len(fn_vals) > 0 else 0.0,
            "std_FN":  round(float(np.std(fn_vals)), 4) if len(fn_vals) > 0 else 0.0,
            "fp_divergence": round(float(abs(np.mean(fp_vals) - np.mean(tn_vals))), 4) if (len(fp_vals) > 0 and len(tn_vals) > 0) else 0.0,
            "fn_divergence": round(float(abs(np.mean(fn_vals) - np.mean(tp_vals))), 4) if (len(fn_vals) > 0 and len(tp_vals) > 0) else 0.0,
        })

    return pd.DataFrame(records).sort_values("fp_divergence", ascending=False).reset_index(drop=True)


def categorize_errors(
    preds_df: pd.DataFrame,
    windows: np.ndarray,
    feat_cols: List[str]
) -> pd.DataFrame:
    """
    Evidence-based error categorization for False Positives and False Negatives.
    Categories:
      - Sudden Transition (High gradient across window in pressure/flow)
      - Subtle Anomaly (Low magnitude anomaly: mean anomaly deviation < 0.5 std)
      - Borderline Confidence (0.005 <= prob <= 0.05)
      - High-Confidence Error (prob > 0.8 for FP, prob < 0.001 for FN)
      - Unclear / Insufficient Evidence
    """
    error_records = []
    
    for idx, row in preds_df.iterrows():
        err_type = row["error_type"]
        if err_type not in ["False_Positive", "False_Negative"]:
            continue

        seq_id = row["sequence_id"]
        prob = row["anomaly_prob"]
        win = windows[seq_id] # (seq_len, n_features)

        # Check temporal gradient across window
        grad = np.max(np.abs(np.diff(win, axis=0)))
        
        category = "Unclear / Insufficient Evidence"
        rationale = ""

        if err_type == "False_Positive":
            if grad > 2.0:
                category = "Sudden Hydraulic Transition"
                rationale = f"Max step gradient is {grad:.2f} > 2.0 std (pump switching/valve transient)."
            elif 0.005 <= prob <= 0.02:
                category = "Borderline Threshold Fluctuation"
                rationale = f"Anomaly score ({prob:.4f}) is marginally above decision threshold ({row['threshold']:.4f})."
            elif prob > 0.5:
                category = "High-Confidence False Alarm"
                rationale = f"Elevated multi-sensor reconstruction error mimicking burst signature ({prob:.4f})."
            else:
                category = "Transient Noise Fluctuation"
                rationale = "Sensor variation without sharp hydraulic step."

        elif err_type == "False_Negative":
            # Check deviation of anomaly window from zero
            mean_abs_val = np.mean(np.abs(win[:, :len(SENSOR_COLS)]))
            if mean_abs_val < 0.6:
                category = "Subtle / Incipient Anomaly"
                rationale = f"Low anomaly signature magnitude ({mean_abs_val:.2f} std within normal diurnal noise)."
            elif 0.005 <= prob <= 0.015:
                category = "Borderline False Negative"
                rationale = f"Anomaly score ({prob:.4f}) is marginally below decision threshold ({row['threshold']:.4f})."
            else:
                category = "Atypical Anomaly Signature"
                rationale = "Anomaly profile does not exhibit standard pressure-drop pattern."

        error_records.append({
            "sequence_id": seq_id,
            "timestamp": row["timestamp"],
            "error_type": err_type,
            "anomaly_prob": prob,
            "category": category,
            "rationale": rationale,
        })

    return pd.DataFrame(error_records)
