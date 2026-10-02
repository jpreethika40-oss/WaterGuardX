"""Download and inspect BattLeDIM XLSX files."""
import requests, warnings, os
warnings.filterwarnings('ignore')

BASE = 'https://zenodo.org/api/records/4017659/files'

xlsx_files = {
    '2018_SCADA.xlsx': 'data/raw/2018_SCADA.xlsx',
    '2019_SCADA.xlsx': 'data/raw/2019_SCADA.xlsx',
}

for fname, dest in xlsx_files.items():
    url = f'{BASE}/{fname}/content'
    print(f"Downloading {fname}...")
    r = requests.get(url, timeout=120, verify=False)
    print(f"  status={r.status_code} bytes={len(r.content)}")
    if r.status_code == 200:
        with open(dest, 'wb') as f:
            f.write(r.content)
        print(f"  -> saved {dest}")
