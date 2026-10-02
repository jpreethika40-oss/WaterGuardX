"""Download BATADAL dataset files."""
import requests
import warnings
import sys
import os

warnings.filterwarnings('ignore')

def download(url, dest):
    try:
        r = requests.get(url, timeout=30, verify=False, allow_redirects=True)
        print(f"  status={r.status_code} bytes={len(r.content)}")
        if r.status_code == 200 and len(r.content) > 500:
            with open(dest, 'wb') as f:
                f.write(r.content)
            print(f"  saved -> {dest}")
            return True
        else:
            print(f"  content preview: {r.content[:200]}")
            return False
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {e}")
        return False

# GitHub API to find download URLs
print("=== Checking GitHub API ===")
r = requests.get('https://api.github.com/repos/scy-phy/BATADAL/contents/data',
                 timeout=15, verify=False)
print(f"API status: {r.status_code}")
print(r.text[:1000])

print("\n=== Trying direct GitHub raw URLs ===")
files = {
    'BATADAL_dataset03.csv': 'data/raw/BATADAL_train1.csv',
    'BATADAL_dataset04.csv': 'data/raw/BATADAL_train2.csv',
    'BATADAL_test_dataset.csv': 'data/raw/BATADAL_test.csv',
}
base = 'https://github.com/scy-phy/BATADAL/raw/master/data/'
for fname, dest in files.items():
    url = base + fname
    print(f"\nTrying: {url}")
    download(url, dest)

print("\n=== Checking batadal.net directly ===")
for fname, dest in files.items():
    url = f'http://www.batadal.net/images/{fname}'
    print(f"\nTrying: {url}")
    download(url, dest)
