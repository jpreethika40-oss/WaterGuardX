"""
Validate BattLeDIM anomaly labelling strategy.

The leakage files give the actual leakage flow rate (m3/h) at each pipe.
Strategy: a timestep is anomalous if total leakage flow > threshold.
This represents a detectable pipe burst that affects sensor readings.

We focus on 2018 data only (2019 has leakages from day 1, no clean baseline).
In 2018, the first leakage starts 2018-01-08 13:30.
So we have ~8 days of clean normal data before any leakage.

We will use a DIFFERENT labelling approach:
- Use the SCADA pressure/flow/level sensors as features
- Use the leakage flow magnitude as the anomaly signal
- Define anomaly = total leakage flow > 1.0 m3/h (detectable burst)
- This gives us a meaningful binary classification problem

Let's analyze the class distribution under different thresholds.
"""
import pandas as pd
import numpy as np

def load_leakages_2018():
    df = pd.read_csv('data/raw/2018_Leakages.csv', sep=';', on_bad_lines='skip', low_memory=False)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df

df_leak = load_leakages_2018()
leak_cols = [c for c in df_leak.columns if c != 'Timestamp']
df_leak['total_leak'] = df_leak[leak_cols].sum(axis=1)

print("=== Leakage Flow Distribution ===")
print(f"Total rows: {len(df_leak)}")
print(f"Date range: {df_leak['Timestamp'].min()} to {df_leak['Timestamp'].max()}")
print(f"\nTotal leakage flow statistics:")
print(df_leak['total_leak'].describe())

print("\n=== Class distribution under different thresholds ===")
for thresh in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    anomaly = (df_leak['total_leak'] > thresh).astype(int)
    n_anom = anomaly.sum()
    n_norm = len(anomaly) - n_anom
    print(f"  threshold={thresh:.1f}: normal={n_norm} ({n_norm/len(anomaly)*100:.1f}%), "
          f"anomaly={n_anom} ({n_anom/len(anomaly)*100:.1f}%)")

print("\n=== First leakage event ===")
first_leak = df_leak[df_leak['total_leak'] > 0]['Timestamp'].min()
print(f"First non-zero leakage: {first_leak}")
print(f"Clean normal period: 2018-01-01 to {first_leak}")
clean_rows = (df_leak['Timestamp'] < first_leak).sum()
print(f"Clean normal rows: {clean_rows} ({clean_rows * 5 / 60:.1f} hours)")

print("\n=== Leakage events by pipe (abrupt vs incipient) ===")
# From config: abrupt leakages start immediately at full flow
# incipient leakages grow gradually to peak
abrupt_pipes = ['p673', 'p538', 'p866', 'p183', 'p158', 'p369']
incipient_pipes = ['p257', 'p461', 'p232', 'p427', 'p628', 'p31', 'p654']

print("Abrupt leakages (immediate full flow):")
for p in abrupt_pipes:
    if p in df_leak.columns:
        nz = (df_leak[p] > 0).sum()
        if nz > 0:
            print(f"  {p}: {nz} rows, max={df_leak[p].max():.2f} m3/h")

print("Incipient leakages (gradual growth):")
for p in incipient_pipes:
    if p in df_leak.columns:
        nz = (df_leak[p] > 0).sum()
        if nz > 0:
            print(f"  {p}: {nz} rows, max={df_leak[p].max():.2f} m3/h")

# Check if p257 (the always-on leakage) is small enough to be "background"
print(f"\np257 (background leakage) stats:")
print(df_leak['p257'].describe())
print(f"p427 (background leakage) stats:")
print(df_leak['p427'].describe())

# Proposed strategy: exclude background leakages (p257, p427, p654, p810)
# These are very small (< 7 m3/h) and represent infrastructure baseline
# Focus on BURST events (abrupt + large incipient)
background_pipes = ['p257', 'p427', 'p654', 'p810']
burst_pipes = [c for c in leak_cols if c not in background_pipes]
df_leak['burst_leak'] = df_leak[burst_pipes].sum(axis=1)

print(f"\n=== Burst-only leakage (excluding background pipes) ===")
for thresh in [0.0, 0.5, 1.0, 2.0]:
    anomaly = (df_leak['burst_leak'] > thresh).astype(int)
    n_anom = anomaly.sum()
    n_norm = len(anomaly) - n_anom
    print(f"  threshold={thresh:.1f}: normal={n_norm} ({n_norm/len(anomaly)*100:.1f}%), "
          f"anomaly={n_anom} ({n_anom/len(anomaly)*100:.1f}%)")
