import json
with open("events.json", "r") as f:
    data = json.load(f)
# Compact JSON
out = json.dumps(data, separators=(',', ':'))
print(len(out))