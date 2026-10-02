"""Extract pump sensor data from GitHub notebook or find direct CSV."""
import requests, warnings, json
warnings.filterwarnings('ignore')

# Check the notebook that has pump sensor data
print("=== Checking SofiyaPovargo pump sensor notebook ===")
r = requests.get('https://api.github.com/repos/SofiyaPovargo/pump_sensor_data/contents',
                 timeout=10, verify=False)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    for item in r.json():
        print(f"  {item['type']}: {item['name']} ({item.get('size',0)} bytes)")
        if item['name'].endswith('.csv'):
            print(f"  -> CSV found! download_url: {item.get('download_url','?')}")

# Search more broadly for pump sensor CSV on GitHub
print("\n=== Broader search for pump sensor CSV ===")
r2 = requests.get('https://api.github.com/search/code?q=sensor_00+BROKEN+NORMAL+filename:sensor.csv',
                  timeout=10, verify=False)
print(f"Code search status: {r2.status_code}")
if r2.status_code == 200:
    data = r2.json()
    print(f"Total: {data.get('total_count',0)}")
    for item in data.get('items',[])[:5]:
        print(f"  {item['repository']['full_name']}: {item['name']}")
        print(f"    {item.get('html_url','?')}")

# Try the anseldsouza water pump dataset
print("\n=== anseldsouza water pump sensor ===")
r3 = requests.get('https://api.github.com/search/repositories?q=anseldsouza+water+pump+sensor',
                  timeout=10, verify=False)
data3 = r3.json()
for item in data3.get('items',[])[:3]:
    print(f"  {item['full_name']}")
    rc = requests.get(f"https://api.github.com/repos/{item['full_name']}/contents",
                      timeout=10, verify=False)
    if rc.status_code == 200:
        for entry in rc.json()[:10]:
            print(f"    {entry['type']}: {entry['name']} ({entry.get('size',0)} bytes)")
            if entry.get('download_url'):
                print(f"      -> {entry['download_url']}")
