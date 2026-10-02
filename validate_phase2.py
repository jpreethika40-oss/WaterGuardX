"""Phase 2 validation — verify all outputs are correct."""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from torch.utils.data import DataLoader

from data_loader import load_raw
from preprocessing import run_preprocessing
from datasets import WaterSensorDataset, MaskedSensorDataset
from config import SENSOR_COLS, TARGET_COL, SEQUENCE_LENGTH

PASS = "[PASS]"
FAIL = "[FAIL]"

checks = []

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    checks.append((status, name, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ''))

print("=" * 60)
print("PHASE 2 VALIDATION")
print("=" * 60)

# ── 1. Dataset loads ──────────────────────────────────────────────────────────
print("\n[1] Dataset loading")
df = load_raw()
check("Dataset loads", df is not None)
check("Shape correct", df.shape == (105120, 14), f"{df.shape}")
check("No missing values", df.isnull().sum().sum() == 0)
check("No duplicates", df.duplicated().sum() == 0)
check("Timestamp column", 'Timestamp' in df.columns)
check("Label column", TARGET_COL in df.columns)
check("All sensors present", all(c in df.columns for c in SENSOR_COLS))

# ── 2. Target ─────────────────────────────────────────────────────────────────
print("\n[2] Target analysis")
check("Binary labels", set(df[TARGET_COL].unique()) == {0, 1})
anomaly_pct = df[TARGET_COL].mean() * 100
check("Anomaly rate 10-20%", 10 <= anomaly_pct <= 20, f"{anomaly_pct:.2f}%")
check("Normal class exists", (df[TARGET_COL] == 0).sum() > 0)
check("Anomaly class exists", (df[TARGET_COL] == 1).sum() > 0)

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
print("\n[3] Preprocessing")
train, val, test, scaler, feat_cols = run_preprocessing(df, save=False)
check("Train split exists", len(train) > 0, f"{len(train)} rows")
check("Val split exists",   len(val) > 0,   f"{len(val)} rows")
check("Test split exists",  len(test) > 0,  f"{len(test)} rows")
check("No temporal leakage", train['Timestamp'].max() < val['Timestamp'].min())
check("Val before test",     val['Timestamp'].max() < test['Timestamp'].min())
total = len(train) + len(val) + len(test)
check("Splits cover full dataset", total == len(df), f"{total} == {len(df)}")

# ── 4. Scaling ────────────────────────────────────────────────────────────────
print("\n[4] Scaling")
train_means = train[SENSOR_COLS].mean()
check("Train sensors near zero mean", abs(train_means.mean()) < 0.1,
      f"mean of means = {train_means.mean():.4f}")
train_stds = train[SENSOR_COLS].std()
check("Train sensors near unit std", abs(train_stds.mean() - 1.0) < 0.1,
      f"mean of stds = {train_stds.mean():.4f}")

# ── 5. Feature engineering ────────────────────────────────────────────────────
print("\n[5] Feature engineering")
check("Temporal features added", all(c in feat_cols for c in
      ['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow']))
check("Feature count correct", len(feat_cols) == 16, f"{len(feat_cols)}")

# ── 6. Dataset class ──────────────────────────────────────────────────────────
print("\n[6] PyTorch Dataset")
ds_train = WaterSensorDataset(train, feat_cols)
check("Train dataset created", len(ds_train) > 0, f"{len(ds_train)} windows")
check("Window shape correct", ds_train.windows.shape[1:] == (SEQUENCE_LENGTH, 16),
      f"{ds_train.windows.shape[1:]}")
x, y = ds_train[0]
check("Sample x shape", x.shape == (SEQUENCE_LENGTH, 16), f"{x.shape}")
check("Sample y is scalar", y.dim() == 0)
check("Class weights computed", ds_train.class_weights is not None)

# ── 7. Masked dataset ─────────────────────────────────────────────────────────
print("\n[7] Masked Dataset (SSL)")
mds = MaskedSensorDataset(train, feat_cols)
check("Masked dataset created", len(mds) > 0, f"{len(mds)} windows")
mx, orig, mask = mds[0]
check("Masked shape correct", mx.shape == (SEQUENCE_LENGTH, 16))
check("Mask ratio ~15%", 0.10 <= mask.mean().item() <= 0.20,
      f"{mask.mean().item():.3f}")
check("Only normal samples", len(mds) <= (train[TARGET_COL] == 0).sum())

# ── 8. Saved files ────────────────────────────────────────────────────────────
print("\n[8] Saved files")
required_files = [
    'data/processed/train.csv',
    'data/processed/val.csv',
    'data/processed/test.csv',
    'data/processed/scaler.pkl',
    'data/processed/preprocessing_meta.json',
    'results/data_summary/data_quality_report.json',
    'figures/eda/01_class_distribution.png',
    'figures/eda/02_sensor_distributions.png',
    'figures/eda/03_correlation_heatmap.png',
    'figures/eda/04_timeseries_overview.png',
    'figures/eda/05_anomaly_timeline.png',
    'figures/eda/06_normal_vs_anomaly_detail.png',
    'figures/eda/07_rolling_statistics.png',
    'figures/eda/08_split_distributions.png',
]
for f in required_files:
    check(f"File exists: {Path(f).name}", Path(f).exists())

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for s, _, _ in checks if s == PASS)
failed = sum(1 for s, _, _ in checks if s == FAIL)
print(f"RESULT: {passed}/{len(checks)} checks passed, {failed} failed")
if failed == 0:
    print("Phase 2 validation: ALL CHECKS PASSED")
else:
    print("Phase 2 validation: SOME CHECKS FAILED")
    for s, name, detail in checks:
        if s == FAIL:
            print(f"  FAILED: {name} — {detail}")
