import json
with open('events.json', 'r') as f:
    data = json.load(f)
print(len(data['events']))
print(json.dumps(data['events'][0], indent=2))
print(json.dumps(data['events'][-1], indent=2))