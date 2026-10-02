"""Inspect BattLeDIM XLSX files."""
import pandas as pd

for year in ['2018', '2019']:
    path = f'data/raw/{year}_SCADA.xlsx'
    print(f"\n{'='*60}")
    print(f"FILE: {path}")
    print(f"{'='*60}")
    xl = pd.ExcelFile(path)
    print(f"Sheets: {xl.sheet_names}")
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, nrows=3)
        print(f"\n  Sheet: {sheet}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns[:10])}{'...' if len(df.columns)>10 else ''}")
        print(f"  Dtypes: {dict(list(df.dtypes.items())[:5])}")
        print(f"  Sample row 0: {df.iloc[0].values[:5]}")
        # Full shape
        df_full = pd.read_excel(path, sheet_name=sheet)
        print(f"  Full shape: {df_full.shape}")
        print(f"  Missing: {df_full.isnull().sum().sum()}")
        print(f"  First col sample: {df_full.iloc[:3, 0].values}")
