"""Run full EDA and generate all figures."""
import sys
sys.path.insert(0, 'src')

from data_loader import load_raw, dataset_summary, print_summary
from preprocessing import run_preprocessing
from visualization import (
    plot_class_distribution,
    plot_sensor_distributions,
    plot_correlation_heatmap,
    plot_timeseries_overview,
    plot_anomaly_timeline,
    plot_normal_vs_anomaly_detail,
    plot_rolling_stats,
    plot_split_distributions,
)
import json
import numpy as np
from pathlib import Path

print("Loading data...")
df = load_raw()
s = dataset_summary(df)
print_summary(s)

print("\nRunning preprocessing...")
train, val, test, scaler, feat_cols = run_preprocessing(df, save=True)

print("\nGenerating EDA figures...")

print("  1/8 Class distribution")
plot_class_distribution(df)

print("  2/8 Sensor distributions")
plot_sensor_distributions(df)

print("  3/8 Correlation heatmap")
plot_correlation_heatmap(df)

print("  4/8 Time-series overview (this may take a moment...)")
plot_timeseries_overview(df)

print("  5/8 Anomaly timeline")
plot_anomaly_timeline(df)

print("  6/8 Normal vs anomaly detail")
plot_normal_vs_anomaly_detail(df)

print("  7/8 Rolling statistics")
plot_rolling_stats(df, sensor='n1', window=48)

print("  8/8 Split distributions")
plot_split_distributions(train, val, test)

# Save data quality report
print("\nSaving data quality report...")
Path('results/data_summary').mkdir(parents=True, exist_ok=True)

quality_report = {
    'dataset': 'Synthetic Water Sensor Dataset (BattLeDIM-derived)',
    'source': 'Generated from BattLeDIM 2018 SCADA sensor statistics (Zenodo ID: 4017659)',
    'generation_method': 'Statistical simulation from real sensor means/stds with realistic temporal patterns and injected anomaly events',
    'total_rows': int(len(df)),
    'total_features': int(len(feat_cols)),
    'sensor_cols': feat_cols,
    'timestamp_range': {'start': str(df['Timestamp'].min()), 'end': str(df['Timestamp'].max())},
    'sampling_frequency': '5 minutes',
    'missing_values': int(df.isnull().sum().sum()),
    'duplicate_rows': int(df.duplicated().sum()),
    'class_distribution': {
        'normal': int((df['label'] == 0).sum()),
        'anomaly': int((df['label'] == 1).sum()),
        'normal_pct': float((df['label'] == 0).mean() * 100),
        'anomaly_pct': float((df['label'] == 1).mean() * 100),
    },
    'anomaly_events': 15,
    'anomaly_types': ['pressure_drop', 'flow_spike', 'level_anomaly', 'combined'],
    'splits': {
        'train': {'rows': len(train), 'anomaly_pct': float(train['label'].mean() * 100),
                  'period': f"{train['Timestamp'].min()} to {train['Timestamp'].max()}"},
        'val':   {'rows': len(val),   'anomaly_pct': float(val['label'].mean() * 100),
                  'period': f"{val['Timestamp'].min()} to {val['Timestamp'].max()}"},
        'test':  {'rows': len(test),  'anomaly_pct': float(test['label'].mean() * 100),
                  'period': f"{test['Timestamp'].min()} to {test['Timestamp'].max()}"},
    },
    'preprocessing': {
        'scaler': 'StandardScaler (fit on train only)',
        'temporal_features': ['sin_hour', 'cos_hour', 'sin_dow', 'cos_dow'],
        'split_strategy': 'Chronological (no random shuffling)',
        'leakage_prevention': 'Scaler fitted on training data only; splits are non-overlapping time periods',
    },
    'data_quality_issues': {
        'missing_values': 'None',
        'duplicates': 'None',
        'constant_features': 'None',
        'outliers': 'Anomaly events are intentional; clipped to 1.5x sensor range',
        'class_imbalance': '86.6% normal / 13.4% anomaly — handled via class weighting in training',
    },
}

with open('results/data_summary/data_quality_report.json', 'w') as f:
    json.dump(quality_report, f, indent=2)

print("\nAll EDA complete.")
print("\nFiles created:")
print("  figures/eda/01_class_distribution.png")
print("  figures/eda/02_sensor_distributions.png")
print("  figures/eda/03_correlation_heatmap.png")
print("  figures/eda/04_timeseries_overview.png")
print("  figures/eda/05_anomaly_timeline.png")
print("  figures/eda/06_normal_vs_anomaly_detail.png")
print("  figures/eda/07_rolling_statistics.png")
print("  figures/eda/08_split_distributions.png")
print("  results/data_summary/dataset_summary.json")
print("  results/data_summary/data_quality_report.json")
print("  data/processed/train.csv")
print("  data/processed/val.csv")
print("  data/processed/test.csv")
print("  data/processed/scaler.pkl")
print("  data/processed/preprocessing_meta.json")
