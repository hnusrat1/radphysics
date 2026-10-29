import json

try:
    with open("output.json", "r") as f:
        data = json.load(f)
    
    n_events = len(data['events'])
    if n_events != 200:
        print(f"ERROR: Expected 200 events, found {n_events}")
    else:
        # Output compact JSON
        print(json.dumps(data, separators=(',', ':')))
except Exception as e:
    print(f"Error: {e}")