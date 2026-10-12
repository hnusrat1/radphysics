import json

with open('result.json', 'r') as f:
    data = json.load(f)

print(json.dumps(data, indent=2))