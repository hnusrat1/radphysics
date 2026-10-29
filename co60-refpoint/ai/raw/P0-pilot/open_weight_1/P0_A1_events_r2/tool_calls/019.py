import json
with open('/tmp/output.json', 'r') as f:
    data = json.load(f)
# Check schema compliance
required_keys = {'process', 'e_in_MeV', 'e_out_MeV', 'cos_scatter', 't_e_MeV', 'cos_electron', 't_positron_MeV'}
for i, e in enumerate(data['events']):
    if set(e.keys()) != required_keys:
        print(f"Event {i} keys mismatch: {set(e.keys())}")
    if e['process'] not in ('compton', 'photoelectric', 'pair', 'rayleigh'):
        print(f"Event {i} invalid process: {e['process']}")
    if not isinstance(e['e_in_MeV'], (int, float)):
        print(f"Event {i} e_in type")
    if e['e_out_MeV'] is not None and not isinstance(e['e_out_MeV'], (int, float)):
        print(f"Event {i} e_out type")
    if e['cos_scatter'] is not None and not isinstance(e['cos_scatter'], (int, float)):
        print(f"Event {i} cos_scatter type")
    if not isinstance(e['t_e_MeV'], (int, float)):
        print(f"Event {i} t_e type")
    if e['cos_electron'] is not None and not isinstance(e['cos_electron'], (int, float)):
        print(f"Event {i} cos_electron type")
    if e['t_positron_MeV'] is not None and not isinstance(e['t_positron_MeV'], (int, float)):
        print(f"Event {i} t_positron type")
print("Validation complete")
print(f"Total events: {len(data['events'])}")
# Count processes
from collections import Counter
cnt = Counter(e['process'] for e in data['events'])
print("Process counts:", dict(cnt))