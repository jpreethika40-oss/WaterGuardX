"""Preprocessing pipeline for water sensor data."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config import (SENSOR_COLS, TARGET_COL, TIMESTAMP_COL,
                    TRAIN_END, VAL_END, SCALER_TYPE, SEED, DATA_PROC)


# ── Chronological split ───────────────────────────────────────────────────────

def chronological_split(df: pd.DataFrame):
    """Split into train / val / test by timestamp (no leakage)."""
    train = df[df[TIMESTAMP_COL] <= TRAIN_END].copy()
    val   = df[(df[TIMESTAMP_COL] > TRAIN_END) &
               (df[TIMESTAMP_COL] <= VAL_END)].copy()
    test  = df[df[TIMESTAMP_COL] > VAL_END].copy()
    return train, val, test


# ── Scaler ────────────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame, cols=SENSOR_COLS):
    """Fit scaler on training data only."""
    if SCALER_TYPE == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    scaler.fit(train_df[cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler, cols=SENSOR_COLS) -> pd.DataFrame:
    out = df.copy()
    out[cols] = scaler.transform(df[cols])
    return out


# ── Feature engineering ───────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day and day-of-week cyclical features."""
    out = df.copy()
    hour = out[TIMESTAMP_COL].dt.hour + out[TIMESTAMP_COL].dt.minute / 60.0
    dow  = out[TIMESTAMP_COL].dt.dayofweek
    out['sin_hour'] = np.sin(2 * np.pi * hour / 24)
    out['cos_hour'] = np.cos(2 * np.pi * hour / 24)
    out['sin_dow']  = np.sin(2 * np.pi * dow / 7)
    out['cos_dow']  = np.cos(2 * np.pi * dow / 7)
    return out


def add_rolling_features(df: pd.DataFrame, cols=SENSOR_COLS,
                          windows=(12, 48)) -> pd.DataFrame:
    """Add rolling mean and std for each sensor."""
    out = df.copy()
    for w in windows:
        for col in cols:
            out[f'{col}_rmean_{w}'] = (
                out[col].rolling(w, min_periods=1).mean()
            )
            out[f'{col}_rstd_{w}'] = (
                out[col].rolling(w, min_periods=1).std().fillna(0)
            )
    return out


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(df: pd.DataFrame, save=True):
    """
    Full preprocessing pipeline.
    Returns: train_df, val_df, test_df, scaler, feature_cols
    """
    # 1. Sort chronologically (already done in loader, but ensure)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # 2. No missing values or duplicates in this dataset (verified)

    # 3. Chronological split BEFORE any fitting
    train, val, test = chronological_split(df)

    # 4. Fit scaler on training data only
    scaler = fit_scaler(train)

    # 5. Scale all splits
    train = apply_scaler(train, scaler)
    val   = apply_scaler(val,   scaler)
    test  = apply_scaler(test,  scaler)

    # 6. Add temporal features (no leakage risk — derived from timestamp)
    train = add_temporal_features(train)
    val   = add_temporal_features(val)
    test  = add_temporal_features(test)

    # 7. Feature columns for models
    temporal_feats = ['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow']
    feature_cols   = SENSOR_COLS + temporal_feats

    if save:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        train.to_csv(DATA_PROC / 'train.csv', index=False)
        val.to_csv(DATA_PROC / 'val.csv',     index=False)
        test.to_csv(DATA_PROC / 'test.csv',   index=False)
        joblib.dump(scaler, DATA_PROC / 'scaler.pkl')
        import json
        meta = {
            'feature_cols':   feature_cols,
            'sensor_cols':    SENSOR_COLS,
            'target_col':     TARGET_COL,
            'timestamp_col':  TIMESTAMP_COL,
            'scaler_type':    SCALER_TYPE,
            'train_end':      TRAIN_END,
            'val_end':        VAL_END,
            'train_rows':     len(train),
            'val_rows':       len(val),
            'test_rows':      len(test),
            'train_anomaly_pct': float(train[TARGET_COL].mean() * 100),
            'val_anomaly_pct':   float(val[TARGET_COL].mean() * 100),
            'test_anomaly_pct':  float(test[TARGET_COL].mean() * 100),
        }
        with open(DATA_PROC / 'preprocessing_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved processed splits to {DATA_PROC}")

    return train, val, test, scaler, feature_cols


if __name__ == '__main__':
    from data_loader import load_raw
    df = load_raw()
    train, val, test, scaler, feat_cols = run_preprocessing(df)
    print(f"Train: {len(train)} rows | "
          f"anomaly={train[TARGET_COL].mean()*100:.1f}%")
    print(f"Val:   {len(val)} rows | "
          f"anomaly={val[TARGET_COL].mean()*100:.1f}%")
    print(f"Test:  {len(test)} rows | "
          f"anomaly={test[TARGET_COL].mean()*100:.1f}%")
    print(f"Features: {feat_cols}")
