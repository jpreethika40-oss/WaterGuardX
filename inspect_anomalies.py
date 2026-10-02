"""Deep inspection of leakage labels and anomaly structure."""
import pandas as pd
import numpy as np

def load_leakages(path):
    df = pd.read_csv(path, sep=';', on_bad_lines='skip', low_memory=False)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # Convert comma-decimal columns to float
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df

for year in ['2018', '2019']:
    print(f"\n{'='*60}")
    print(f"LEAKAGE ANALYSIS {year}")
    print(f"{'='*60}")
    df = load_leakages(f'data/raw/{year}_Leakages.csv')
    
    # Binary anomaly: any pipe has non-zero leakage
    leak_cols = [c for c in df.columns if c != 'Timestamp']
    df['any_leak'] = (df[leak_cols] > 0).any(axis=1).astype(int)
    
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"Sampling: 5-minute intervals")
    print(f"\nClass distribution:")
    vc = df['any_leak'].value_counts()
    print(f"  Normal (0): {vc.get(0,0)} ({vc.get(0,0)/len(df)*100:.2f}%)")
    print(f"  Anomaly (1): {vc.get(1,0)} ({vc.get(1,0)/len(df)*100:.2f}%)")
    
    print(f"\nPer-pipe leakage summary:")
    for col in leak_cols:
        nz = (df[col] > 0).sum()
        if nz > 0:
            print(f"  {col}: {nz} rows ({nz/len(df)*100:.2f}%), "
                  f"max={df[col].max():.3f}, mean_when_active={df[col][df[col]>0].mean():.3f}")
    
    # Leakage periods
    print(f"\nLeakage event periods:")
    in_leak = False
    start = None
    events = []
    for _, row in df.iterrows():
        if row['any_leak'] == 1 and not in_leak:
            in_leak = True
            start = row['Timestamp']
        elif row['any_leak'] == 0 and in_leak:
            in_leak = False
            events.append((start, row['Timestamp']))
    if in_leak:
        events.append((start, df['Timestamp'].iloc[-1]))
    
    for i, (s, e) in enumerate(events[:10]):
        dur = (e - s).total_seconds() / 3600
        print(f"  Event {i+1}: {s} -> {e} ({dur:.1f} hours)")
    if len(events) > 10:
        print(f"  ... and {len(events)-10} more events")
    print(f"  Total events: {len(events)}")
