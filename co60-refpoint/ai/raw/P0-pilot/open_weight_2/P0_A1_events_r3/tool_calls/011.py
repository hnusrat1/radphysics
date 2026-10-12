import json

# Load and verify final data
with open('photon_events.json', 'r') as f:
    data = json.load(f)

# Output exactly as needed
print(json.dumps(data, indent=2))
