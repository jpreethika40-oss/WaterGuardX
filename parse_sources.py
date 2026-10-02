"""Parse search results to find BATADAL download links."""
import requests, warnings, json
warnings.filterwarnings('ignore')

# Parse Zenodo results
print("=== ZENODO RESULTS ===")
r = requests.get('https://zenodo.org/api/records?q=BATADAL&size=5', timeout=15, verify=False)
data = r.json()
for hit in data['hits']['hits']:
    print(f"\nTitle: {hit.get('metadata',{}).get('title','?')}")
    print(f"ID: {hit.get('id')}")
    for f in hit.get('files', []):
        print(f"  File: {f.get('key')} -> {f.get('links',{}).get('self','?')}")

# Parse GitHub search
print("\n=== GITHUB REPOS ===")
r2 = requests.get('https://api.github.com/search/repositories?q=BATADAL+water&sort=stars',
                  timeout=15, verify=False)
data2 = r2.json()
for item in data2.get('items', [])[:5]:
    print(f"\nRepo: {item['full_name']} stars={item['stargazers_count']}")
    print(f"  URL: {item['html_url']}")
    # Check contents
    api = f"https://api.github.com/repos/{item['full_name']}/contents"
    rc = requests.get(api, timeout=10, verify=False)
    if rc.status_code == 200:
        for entry in rc.json()[:8]:
            print(f"  {entry['type']}: {entry['name']}")
