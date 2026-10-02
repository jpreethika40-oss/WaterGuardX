"""Download pump sensor dataset."""
import requests, warnings, json
warnings.filterwarnings('ignore')

# Check if there's a GitHub mirror of the pump sensor data
print("=== Searching GitHub for pump sensor data mirrors ===")
r = requests.get('https://api.github.com/search/repositories?q=pump+sensor+data+nphantawee&sort=stars',
                 timeout=10, verify=False)
data = r.json()
print(f"Results: {data.get('total_count',0)}")
for item in data.get('items', [])[:5]:
    print(f"  {item['full_name']} stars={item['stargazers_count']}")
    # Check contents
    rc = requests.get(f"https://api.github.com/repos/{item['full_name']}/contents",
                      timeout=10, verify=False)
    if rc.status_code == 200:
        for entry in rc.json()[:10]:
            print(f"    {entry['type']}: {entry['name']} ({entry.get('size',0)} bytes)")

# Try Kaggle dataset download API
print("\n=== Trying Kaggle dataset download ===")
# The Kaggle API requires authentication, but let's check what's available
r2 = requests.get('https://www.kaggle.com/api/v1/datasets/nphantawee/pump-sensor-data/download',
                  timeout=15, verify=False)
print(f"Download status: {r2.status_code}")
print(f"Headers: {dict(r2.headers)}")
print(f"Content preview: {r2.content[:200]}")

# Check Zenodo for pump sensor
print("\n=== Zenodo pump sensor ===")
r3 = requests.get('https://zenodo.org/api/records?q=pump+sensor+BROKEN+NORMAL&size=5',
                  timeout=15, verify=False)
data3 = r3.json()
for hit in data3.get('hits',{}).get('hits',[]):
    print(f"  {hit.get('metadata',{}).get('title','?')}")
    for f in hit.get('files',[])[:3]:
        print(f"    {f.get('key')} -> {f.get('links',{}).get('self','?')[:60]}")
