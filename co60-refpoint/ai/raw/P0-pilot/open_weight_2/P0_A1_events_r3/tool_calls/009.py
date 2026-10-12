import json

# Read and verify the JSON
with open('photon_events.json', 'r') as f:
    data = json.load(f)

print("JSON loaded successfully")
print(f"Number of events: {len(data['events'])}")
print(f"Method string length: {len(data['method'])}")

# Check first event structure
first_event = data['events'][0]
print("\nFirst event keys:", list(first_event.keys()))
print("First event values:")
for k, v in first_event.items():
    print(f"  {k}: {v} (type: {type(v).__name__})")

# Verify null handling
compton_event = [e for e in data['events'] if e['process'] == 'compton'][0]
print("\nCompton event example:")
for k, v in compton_event.items():
    print(f"  {k}: {v} ({type(v).__name__})")

pair_event = [e for e in data['events'] if e['process'] == 'pair'][0]
print("\nPair event example:")
for k, v in pair_event.items():
    print(f"  {k}: {v} ({type(v).__name__})")

photo_event = [e for e in data['events'] if e['process'] == 'photoelectric'][0]
print("\nPhotoelectric event example:")
for k, v in photo_event.items():
    print(f"  {k}: {v} ({type(v).__name__})")

rayleigh_event = [e for e in data['events'] if e['process'] == 'rayleigh'][0]
print("\nRayleigh event example:")
for k, v in rayleigh_event.items():
    print(f"  {k}: {v} ({type(v).__name__})")
