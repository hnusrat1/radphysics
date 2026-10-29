import json

with open("final_events.json","r") as f:
    data = json.load(f)

events = data["events"]
print(f"Number of events: {len(events)}")
print(f"Processes:")
from collections import Counter
print(Counter([e["process"] for e in events]))

# Let's also print a few sample events to check format
for i in range(5):
    print(json.dumps(events[i], indent=2))

# Check that all required fields are present
for i, e in enumerate(events):
    required = ["process","e_in_MeV","e_out_MeV","cos_scatter","t_e_MeV","cos_electron","t_positron_MeV"]
    for r in required:
        if r not in e:
            print(f"Missing {r} in event {i}")
    if e["process"] not in ["compton","photoelectric","pair","rayleigh"]:
        print(f"Invalid process in event {i}: {e['process']}")