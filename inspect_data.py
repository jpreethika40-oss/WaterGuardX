"""Inspect BattLeDIM dataset structure."""
import pandas as pd
import numpy as np
import os

files = {
    'pressures_2018': 'data/raw/2018_SCADA_Pressures.csv',
    'flows_2018':     'data/raw/2018_SCADA_Flows.csv',
    'levels_2018':    'data/raw/2018_SCADA_Levels.csv',
    'demands_2018':   'data/raw/2018_SCADA_Demands.csv',
    'leakages_2018':  'data/raw/2018_Leakages.csv',
    'pressures_2019': 'data/raw/2019_SCADA_Pressures.csv',
    'flows_2019':     'data/raw/2019_SCADA_Flows.csv',
    'levels_2019':    'data/raw/2019_SCADA_Levels.csv',
    'demands_2019':   'data/raw/2019_SCADA_Demands.csv',
    'leakages_2019':  'data/raw/2019_Leakages.csv',
}

for name, path in files.items():
    print(f"\n{'='*60}")
    print(f"FILE: {name} ({path})")
    print(f"{'='*60}")
    try:
        df = pd.read_csv(path, nrows=5)
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"Sample:\n{df.head(3).to_string()}")
        # Full read for shape
        df_full = pd.read_csv(path)
        print(f"Full shape: {df_full.shape}")
        print(f"Missing values: {df_full.isnull().sum().sum()}")
    except Exception as e:
        print(f"ERROR: {e}")

print("\n=== README ===")
with open('data/raw/README.txt') as f:
    print(f.read())
