with open('output.json', 'r') as f:
    data = json.load(f)

print(f"Loaded JSON: method='{data['method'][:50]}...'")
print(f"Number of events: {len(data['events'])}")

# Verify all processes
processes = set(e['process'] for e in data['events'])
print(f"Processes present: {processes}")

# Verify no None issues
for e in data['events'][:3]:
    print(f"\n{e['process']}:")
    for k, v in e.items():
        print(f"  {k}: {v}")

# Check last few
for e in data['events'][-3:]:
    print(f"\n{e['process']}:")
    for k, v in e.items():
        print(f"  {k}: {v}")

# Validate all events exist
assert len(data['events']) == 200
print("\nAll good!")