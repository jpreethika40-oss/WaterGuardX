"""Search for BATADAL dataset across known mirrors."""
import requests
import warnings
warnings.filterwarnings('ignore')

candidates = [
    # Zenodo
    'https://zenodo.org/api/records?q=BATADAL&size=5',
    # Known GitHub repos that host BATADAL
    'https://api.github.com/search/repositories?q=BATADAL+water&sort=stars',
    # Direct known mirrors
    'https://raw.githubusercontent.com/rtaormina/BATADAL/main/data/BATADAL_dataset03.csv',
    'https://raw.githubusercontent.com/rtaormina/BATADAL/master/data/BATADAL_dataset03.csv',
    'https://raw.githubusercontent.com/rtaormina/aeed/master/data/BATADAL_dataset03.csv',
    'https://raw.githubusercontent.com/mitre/caldera/master/data/BATADAL_dataset03.csv',
    # UCI / other repos
    'https://raw.githubusercontent.com/KIOS-Research/BattLeDIM/master/Dataset/BATADAL_dataset03.csv',
]

for url in candidates:
    try:
        r = requests.get(url, timeout=15, verify=False)
        print(f"[{r.status_code}] {url[:80]}")
        if r.status_code == 200:
            print(f"  bytes={len(r.content)}, preview={r.content[:120]}")
    except Exception as e:
        print(f"[ERR] {url[:80]} -> {e}")
