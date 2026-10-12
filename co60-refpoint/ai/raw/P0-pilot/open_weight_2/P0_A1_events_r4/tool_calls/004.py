import json
with open('out.json','r') as f:
    data = json.load(f)
# Compact it
print(json.dumps(data, separators=(',',':')))