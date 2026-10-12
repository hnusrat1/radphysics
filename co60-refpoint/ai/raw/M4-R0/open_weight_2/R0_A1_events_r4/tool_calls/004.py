import json
with open("events.json", "r") as f:
    data = json.load(f)
print(json.dumps(data, indent=2))