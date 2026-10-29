import json
with open('output.json', 'r') as f:
    data = json.load(f)

print(f"Number of events: {len(data['events'])}")
processes = set(e['process'] for e in data['events'])
print(f"Processes present: {processes}")

# Check a few random events
for i, e in enumerate(data['events'][:3]):
    print(f"\nEvent {i+1} ({e['process']}):")
    for k, v in e.items():
        print(f"  {k}: {v}")

# Check last
for i, e in enumerate(data['events'][-2:]):
    print(f"\nEvent {199+i} ({e['process']}):")
    for k, v in e.items():
        print(f"  {k}: {v}")

# Validate schema
for e in data['events']:
    assert e['process'] in ['compton', 'photoelectric', 'pair', 'rayleigh']
    assert isinstance(e['e_in_MeV'], (int, float))
    assert e['e_out_MeV'] is None or isinstance(e['e_out_MeV'], (int, float))
    assert e['cos_scatter'] is None or isinstance(e['cos_scatter'], (int, float))
    assert isinstance(e['t_e_MeV'], (int, float))
    assert e['cos_electron'] is None or isinstance(e['cos_electron'], (int, float))
    assert e['t_positron_MeV'] is None or isinstance(e['t_positron_MeV'], (int, float))

print(f"\nAll {len(data['events'])} events pass validation!")
print(f"Method: {data['method']}")