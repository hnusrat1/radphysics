import json

# We know we have the data in final.json
with open('final.json', 'r') as f:
    data = json.load(f)

# Verify structure
print("Method:", data['method'][:50])
print("Num Events:", len(data['events']))