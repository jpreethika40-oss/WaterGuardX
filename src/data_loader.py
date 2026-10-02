"""Load and inspect the water sensor dataset."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from config import (RAW_DATA_FILE, SENSOR_COLS, TARGET_COL,
                    TIMESTAMP_COL, ANOMALY_LOG)


def load_raw(path=RAW_DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    summary = {
        'shape':          df.shape,
        'n_rows':         len(df),
        'n_cols':         len(df.columns),
        'columns':        list(df.columns),
        'dtypes':         {c: str(t) for c, t in df.dtypes.items()},
        'timestamp_range': {
            'start': str(df[TIMESTAMP_COL].min()),
            'end':   str(df[TIMESTAMP_COL].max()),
        },
        'sampling_freq':  '5 minutes',
        'sensor_cols':    SENSOR_COLS,
        'target_col':     TARGET_COL,
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing':  int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'class_distribution': df[TARGET_COL].value_counts().to_dict(),
        'class_balance': {
            'normal_pct':  float((df[TARGET_COL] == 0).mean() * 100),
            'anomaly_pct': float((df[TARGET_COL] == 1).mean() * 100),
        },
        'sensor_stats': df[SENSOR_COLS].describe().to_dict(),
    }
    return summary


def print_summary(summary: dict) -> None:
    print(f"{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Shape:          {summary['shape']}")
    print(f"Timestamp:      {summary['timestamp_range']['start']} "
          f"to {summary['timestamp_range']['end']}")
    print(f"Sampling:       {summary['sampling_freq']}")
    print(f"Sensors:        {len(summary['sensor_cols'])} "
          f"({summary['sensor_cols']})")
    print(f"Target:         {summary['target_col']}")
    print(f"Missing values: {summary['total_missing']}")
    print(f"Duplicates:     {summary['duplicate_rows']}")
    print(f"\nClass distribution:")
    print(f"  Normal  (0): {summary['class_distribution'].get(0, 0):>6d} "
          f"({summary['class_balance']['normal_pct']:.2f}%)")
    print(f"  Anomaly (1): {summary['class_distribution'].get(1, 0):>6d} "
          f"({summary['class_balance']['anomaly_pct']:.2f}%)")


if __name__ == '__main__':
    df = load_raw()
    s = dataset_summary(df)
    print_summary(s)
    
    # Save summary
    out = Path('../results/data_summary')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'dataset_summary.json', 'w') as f:
        # Convert non-serialisable types
        import json
        def default(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return str(o)
        json.dump(s, f, indent=2, default=default)
    print(f"\nSaved: results/data_summary/dataset_summary.json")
