import json

# Ensure we have the data
with open('result.json', 'r') as f:
    data = json.load(f)

# Minimize the output size to avoid truncation
minified_json = json.dumps(data, separators=(',', ':'))

with open('final.json', 'w') as f:
    f.write(minified_json)

print(f"Length: {len(minified_json)}")
