"""Download BattLeDIM dataset from Zenodo and inspect BATADAL-related repos."""
import requests, warnings, json, os
warnings.filterwarnings('ignore')

BASE = 'https://zenodo.org/api/records/4017659/files'

# Files to download - SCADA sensor data + leakage labels
files_to_get = {
    '2018_SCADA_Pressures.csv': 'data/raw/2018_SCADA_Pressures.csv',
    '2018_SCADA_Flows.csv':     'data/raw/2018_SCADA_Flows.csv',
    '2018_SCADA_Levels.csv':    'data/raw/2018_SCADA_Levels.csv',
    '2018_SCADA_Demands.csv':   'data/raw/2018_SCADA_Demands.csv',
    '2018_Leakages.csv':        'data/raw/2018_Leakages.csv',
    '2019_SCADA_Pressures.csv': 'data/raw/2019_SCADA_Pressures.csv',
    '2019_SCADA_Flows.csv':     'data/raw/2019_SCADA_Flows.csv',
    '2019_SCADA_Levels.csv':    'data/raw/2019_SCADA_Levels.csv',
    '2019_SCADA_Demands.csv':   'data/raw/2019_SCADA_Demands.csv',
    '2019_Leakages.csv':        'data/raw/2019_Leakages.csv',
    'README.txt':               'data/raw/README.txt',
}

for fname, dest in files_to_get.items():
    url = f'{BASE}/{fname}/content'
    print(f"Downloading {fname}...")
    try:
        r = requests.get(url, timeout=60, verify=False)
        print(f"  status={r.status_code} bytes={len(r.content)}")
        if r.status_code == 200 and len(r.content) > 100:
            with open(dest, 'wb') as f:
                f.write(r.content)
            print(f"  -> saved {dest}")
        else:
            print(f"  -> SKIP (too small or error)")
    except Exception as e:
        print(f"  -> ERR: {e}")

# Also check the BATADAL dissertation repo
print("\n=== Checking BATADAL dissertation repo ===")
r = requests.get('https://api.github.com/repos/saijitendra03/water-distribution-cyberattack-detection/contents',
                 timeout=10, verify=False)
print(r.status_code)
if r.status_code == 200:
    for item in r.json():
        print(f"  {item['type']}: {item['name']} ({item.get('size',0)} bytes)")
