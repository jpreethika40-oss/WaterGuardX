"""Download and inspect BattLeDIM configuration."""
import requests, warnings
warnings.filterwarnings('ignore')

BASE = 'https://zenodo.org/api/records/4017659/files'

# Download config
r = requests.get(f'{BASE}/dataset_configuration.yaml/content', timeout=30, verify=False)
print(f"Config status: {r.status_code}")
config_text = r.text
with open('data/raw/dataset_configuration.yaml', 'w') as f:
    f.write(config_text)
print(config_text)
