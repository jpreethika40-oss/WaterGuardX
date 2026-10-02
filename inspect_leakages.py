"""Inspect leakage label files."""
import pandas as pd
import numpy as np

for year in ['2018', '2019']:
    path = f'data/raw/{year}_Leakages.csv'
    print(f"\n{'='*60}")
    print(f"LEAKAGES {year}: {path}")
    print(f"{'='*60}")
    # Try semicolon separator
    try:
        df = pd.read_csv(path, sep=';', on_bad_lines='skip')
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"Sample:\n{df.head(5).to_string()}")
        print(f"Missing: {df.isnull().sum().sum()}")
        # Check non-zero values (actual leakages)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            nonzero = (df[numeric_cols] != 0).any(axis=1)
            print(f"\nRows with any leakage: {nonzero.sum()} / {len(df)}")
            print(f"Leakage rate: {nonzero.mean():.4f}")
            # Show leakage periods
            print(f"\nLeakage columns with non-zero values:")
            for col in numeric_cols:
                nz = (df[col] != 0).sum()
                if nz > 0:
                    print(f"  {col}: {nz} non-zero rows, max={df[col].max():.3f}")
    except Exception as e:
        print(f"ERROR: {e}")
        # Try with error_bad_lines=False
        try:
            df = pd.read_csv(path, sep=';', error_bad_lines=False)
            print(f"Shape (skip bad): {df.shape}")
        except Exception as e2:
            print(f"ERROR2: {e2}")
