"""
WaterGuardX — Application Utilities.
Loads trained models, preprocessing artifacts, drift reference statistics,
and handles data preprocessing and sequence inference.
"""
import json
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Root and project directories
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / 'data'
DATA_PROC = ROOT / 'data' / 'processed'
DATA_RAW = ROOT / 'data' / 'raw'
MODELS_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'figures'

# Constants
SEQUENCE_LENGTH = 48
SENSOR_COLS = ['n1', 'n54', 'n105', 'n163', 'n215', 'n332', 'n458', 'n549', 'p227', 'p235', 'PUMP_1', 'T1']
TEMPORAL_COLS = ['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow']
FEATURE_COLS = SENSOR_COLS + TEMPORAL_COLS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model_architecture(n_features: int = 16, dropout: float = 0.15):
    """Instantiate the SSL Temporal Transformer architecture."""
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from self_supervised import SSLTransformer
    model = SSLTransformer(
        n_features=n_features,
        d_model=32,
        nhead=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=dropout,
        max_seq_len=SEQUENCE_LENGTH + 10
    )
    return model


def load_waterguardx_models() -> Tuple[nn.Module, Optional[nn.Module], str]:
    """
    Load nominal and adapted model checkpoints.
    Returns: (nominal_model, adapted_model, device)
    """
    nominal_path = MODELS_DIR / 'final_model.pt'
    adapted_path = MODELS_DIR / 'final_model_adapted.pt'

    if not nominal_path.exists():
        raise FileNotFoundError(f"Nominal model checkpoint not found at {nominal_path}")

    nominal_model = load_model_architecture()
    nominal_model.load_state_dict(torch.load(nominal_path, map_location=DEVICE))
    nominal_model.to(DEVICE)
    nominal_model.eval()

    adapted_model = None
    if adapted_path.exists():
        adapted_model = load_model_architecture()
        adapted_model.load_state_dict(torch.load(adapted_path, map_location=DEVICE))
        adapted_model.to(DEVICE)
        adapted_model.eval()

    return nominal_model, adapted_model, DEVICE


def load_preprocessing_artifacts() -> Tuple[object, dict]:
    """Load the trained StandardScaler and metadata json."""
    scaler_path = DATA_PROC / 'scaler.pkl'
    meta_path = DATA_PROC / 'preprocessing_meta.json'

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}")
    scaler = joblib.load(scaler_path)

    meta = {}
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

    return scaler, meta


def load_drift_reference() -> Tuple[dict, dict]:
    """Load reference distribution statistics and drift detector config."""
    ref_path = RESULTS_DIR / 'reference_distribution.json'
    cfg_path = RESULTS_DIR / 'drift' / 'detector_config.json'

    ref_stats = {}
    if ref_path.exists():
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref_stats = json.load(f)

    cfg = {"threshold": 0.009148, "feature_cols": FEATURE_COLS}
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

    return ref_stats, cfg


def load_phase10_results() -> dict:
    """Load Phase 10 final evaluation results and model selection details."""
    res_path = RESULTS_DIR / 'final_results.json'
    sel_path = RESULTS_DIR / 'final_model_selection.json'
    thr_path = RESULTS_DIR / 'final_threshold.json'

    results = {}
    if res_path.exists():
        with open(res_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

    selection = {}
    if sel_path.exists():
        with open(sel_path, 'r', encoding='utf-8') as f:
            selection = json.load(f)

    thresholds = {"nominal_threshold": 0.0100, "adapted_threshold": 0.0100}
    if thr_path.exists():
        with open(thr_path, 'r', encoding='utf-8') as f:
            thresholds = json.load(f)

    return {"results": results, "selection": selection, "thresholds": thresholds}


def load_phase11_results() -> dict:
    """Load Phase 11 error analysis and interpretability outputs."""
    err_dir = RESULTS_DIR / 'error_analysis'
    summary_path = err_dir / 'error_summary.csv'
    interp_path = err_dir / 'interpretability_results.csv'
    diverge_path = err_dir / 'sensor_error_divergence.csv'
    fp_path = err_dir / 'false_positives.csv'
    fn_path = err_dir / 'false_negatives.csv'

    out = {}
    if summary_path.exists():
        out['summary'] = pd.read_csv(summary_path)
    if interp_path.exists():
        out['interpretability'] = pd.read_csv(interp_path)
    if diverge_path.exists():
        out['divergence'] = pd.read_csv(diverge_path)
    if fp_path.exists():
        out['false_positives'] = pd.read_csv(fp_path)
    if fn_path.exists():
        out['false_negatives'] = pd.read_csv(fn_path)

    return out


def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Identify timestamp column regardless of exact casing."""
    candidates = ['Timestamp', 'timestamp', 'datetime', 'Datetime', 'date', 'time']
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'time' in c.lower() or 'date' in c.lower():
            return c
    return None


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], Optional[str]]:
    """
    Validate that uploaded dataframe satisfies WaterGuardX requirements.
    Returns: (is_valid, error_messages, timestamp_col_name)
    """
    errors = []
    ts_col = find_timestamp_column(df)
    if ts_col is None:
        errors.append("Missing timestamp column (e.g. 'Timestamp' or 'datetime').")

    missing_sensors = [s for s in SENSOR_COLS if s not in df.columns]
    if missing_sensors:
        errors.append(f"Missing required sensor columns ({len(missing_sensors)}): {', '.join(missing_sensors)}")

    if len(df) < SEQUENCE_LENGTH:
        errors.append(f"Dataset has {len(df)} rows, but at least {SEQUENCE_LENGTH} consecutive rows are required for temporal sequence generation.")

    is_valid = len(errors) == 0
    return is_valid, errors, ts_col


def preprocess_data(df: pd.DataFrame, scaler, ts_col: str) -> pd.DataFrame:
    """
    Preprocess incoming data: parse timestamp, scale sensors if in raw units, derive temporal encodings.
    Matches exact training pipeline.
    """
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors='coerce')
    out = out.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    # Check if data is in raw physical units (e.g. pressure ~ 20-60, tank ~ 3)
    # vs already standardized (mean ~ 0, std ~ 1)
    mean_sensor_level = float(out[SENSOR_COLS].mean().abs().mean())
    if mean_sensor_level > 2.0:
        # Raw units: apply fitted StandardScaler
        out[SENSOR_COLS] = scaler.transform(out[SENSOR_COLS])

    # Derive cyclical temporal features
    hour = out[ts_col].dt.hour + out[ts_col].dt.minute / 60.0
    dow = out[ts_col].dt.dayofweek
    out['sin_hour'] = np.sin(2 * np.pi * hour / 24)
    out['cos_hour'] = np.cos(2 * np.pi * hour / 24)
    out['sin_dow'] = np.sin(2 * np.pi * dow / 7)
    out['cos_dow'] = np.cos(2 * np.pi * dow / 7)

    return out


def generate_inference_sequences(df: pd.DataFrame, ts_col: str, seq_len: int = SEQUENCE_LENGTH, stride: int = 1):
    """
    Generate sliding sequences of shape (N, seq_len, 16) and corresponding timestamps.
    Returns: sequences_np, timestamps_list, ground_truth_labels (if present)
    """
    X = df[FEATURE_COLS].values.astype(np.float32)
    timestamps = df[ts_col].values
    has_labels = 'label' in df.columns

    seqs = []
    ts_aligned = []
    y_aligned = []

    for start in range(0, len(X) - seq_len + 1, stride):
        end = start + seq_len
        seqs.append(X[start:end])
        ts_aligned.append(timestamps[end - 1])
        if has_labels:
            y_aligned.append(int(df['label'].iloc[start:end].iloc[-1]))

    seqs_np = np.array(seqs, dtype=np.float32)
    y_np = np.array(y_aligned, dtype=np.int64) if has_labels else None

    return seqs_np, ts_aligned, y_np


def run_model_inference(model: nn.Module, sequences: np.ndarray, threshold: float = 0.0100, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run PyTorch inference over generated sequences.
    Returns: (probabilities, binary_predictions)
    """
    model.eval()
    dataset = TensorDataset(torch.from_numpy(sequences))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs_list = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            if hasattr(model, 'forward_classify'):
                logits = model.forward_classify(x)
            else:
                logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probs_list.append(probs)

    probs_all = np.concatenate(probs_list, axis=0)
    preds_all = (probs_all >= threshold).astype(int)
    return probs_all, preds_all


def compute_drift_score(incoming_df: pd.DataFrame) -> dict:
    """
    Compute two-sample Kolmogorov-Smirnov test against reference training distribution.
    Uses reference distribution quantiles from Phase 7.
    """
    ref_path = DATA_PROC / 'train.csv'
    ref_cfg = RESULTS_DIR / 'drift' / 'detector_config.json'

    threshold = 0.009148
    if ref_cfg.exists():
        with open(ref_cfg, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            threshold = cfg.get('threshold', threshold)

    if ref_path.exists():
        ref_df = pd.read_csv(ref_path)
        ref_vals = ref_df[FEATURE_COLS].values
    else:
        # Fallback to standard normal reference
        ref_vals = np.random.normal(0, 1, size=(5000, len(FEATURE_COLS)))

    incoming_vals = incoming_df[FEATURE_COLS].values
    per_feature = {}
    ks_vals = []

    for idx, feat in enumerate(FEATURE_COLS):
        stat, pval = stats.ks_2samp(ref_vals[:, idx], incoming_vals[:, idx])
        is_shifted = bool(stat > threshold)
        per_feature[feat] = {
            "ks_statistic": round(float(stat), 5),
            "p_value": round(float(pval), 6),
            "shift_detected": is_shifted
        }
        ks_vals.append(float(stat))

    mean_ks = float(np.mean(ks_vals))
    shift_detected = bool(mean_ks > threshold)

    return {
        "mean_ks": round(mean_ks, 5),
        "threshold": round(threshold, 5),
        "shift_detected": shift_detected,
        "n_shifted_features": sum(1 for v in per_feature.values() if v["shift_detected"]),
        "total_features": len(FEATURE_COLS),
        "per_feature": per_feature
    }


def load_sample_stream(stream_type: str = 'normal', n_rows: int = 1000) -> pd.DataFrame:
    """
    Load pre-packaged realistic test samples for instantaneous app demonstration.
    Uses held-out December 2018 data.
    """
    sample_dir = DATA_DIR / 'sample'
    if stream_type == 'normal':
        sample_path = sample_dir / 'normal_sample.csv'
        if sample_path.exists():
            return pd.read_csv(sample_path).head(n_rows).reset_index(drop=True)
        raw_path = DATA_RAW / 'water_sensor_data.csv'
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            dec_df = df[df['Timestamp'] >= '2018-12-05'].copy()
            return dec_df.head(n_rows).reset_index(drop=True)
    elif stream_type == 'shifted':
        sample_path = sample_dir / 'shifted_sample.csv'
        if sample_path.exists():
            return pd.read_csv(sample_path).head(n_rows).reset_index(drop=True)
        shift_path = DATA_PROC / 'shifted' / 'strong' / 'test_shifted.csv'
        if shift_path.exists():
            df = pd.read_csv(shift_path)
            dec_df = df[df['Timestamp'] >= '2018-12-05'].copy()
            return dec_df.head(n_rows).reset_index(drop=True)

    # Fallback to raw data
    raw_path = DATA_RAW / 'water_sensor_data.csv'
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        return df.iloc[-n_rows:].copy().reset_index(drop=True)

    raise FileNotFoundError("Could not locate sample sensor data files.")
