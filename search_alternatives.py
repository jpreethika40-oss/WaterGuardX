"""Search for SWaT mirrors and Pump Sensor dataset."""
import requests, warnings, json
warnings.filterwarnings('ignore')

print("=== Searching for SWaT public mirrors ===")
swat_candidates = [
    'https://api.github.com/search/repositories?q=SWaT+dataset+anomaly&sort=stars',
    'https://zenodo.org/api/records?q=SWaT+water+treatment&size=5',
]
for url in swat_candidates:
    r = requests.get(url, timeout=15, verify=False)
    print(f"\n[{r.status_code}] {url[:70]}")
    if r.status_code == 200:
        data = r.json()
        if 'items' in data:
            for item in data['items'][:5]:
                print(f"  {item['full_name']} stars={item['stargazers_count']}")
                print(f"    {item['html_url']}")
        elif 'hits' in data:
            for hit in data['hits']['hits'][:3]:
                print(f"  {hit.get('metadata',{}).get('title','?')}")
                for f in hit.get('files',[])[:3]:
                    print(f"    {f.get('key')} -> {f.get('links',{}).get('self','?')[:60]}")

print("\n=== Searching for Pump Sensor dataset ===")
pump_candidates = [
    'https://api.github.com/search/repositories?q=pump+sensor+predictive+maintenance+anomaly&sort=stars',
    'https://zenodo.org/api/records?q=pump+sensor+anomaly+detection&size=5',
]
for url in pump_candidates:
    r = requests.get(url, timeout=15, verify=False)
    print(f"\n[{r.status_code}] {url[:70]}")
    if r.status_code == 200:
        data = r.json()
        if 'items' in data:
            for item in data['items'][:5]:
                print(f"  {item['full_name']} stars={item['stargazers_count']}")
        elif 'hits' in data:
            for hit in data['hits']['hits'][:3]:
                print(f"  {hit.get('metadata',{}).get('title','?')}")

# Try direct Kaggle API (public datasets)
print("\n=== Trying Kaggle API ===")
r = requests.get('https://www.kaggle.com/api/v1/datasets/list?search=pump+sensor',
                 timeout=10, verify=False)
print(f"Kaggle API: {r.status_code}")
