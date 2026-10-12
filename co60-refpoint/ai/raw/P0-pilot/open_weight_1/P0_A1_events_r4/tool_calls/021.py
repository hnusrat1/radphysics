# Let's just read the previous events.json if it exists, or regenerate using a clean script
import os
if os.path.exists('events.json'):
    with open('events.json', 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['events'])} events from file")
else:
    print("No events.json found")