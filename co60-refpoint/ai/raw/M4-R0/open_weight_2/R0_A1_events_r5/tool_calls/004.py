import json

# Verify file content
with open('output.json', 'r') as f:
    data = json.load(f)

print(f"Number of events: {len(data['events'])}")
print(f"First event process: {data['events'][0]['process']}")
print(f"Last event process: {data['events'][-1]['process']}")

# Just to be safe, re-serialize and check size
json_str = json.dumps(data)
print(f"JSON length: {len(json_str)} chars")