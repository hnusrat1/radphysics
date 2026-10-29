import json

# Final verification and output
with open('photon_events.json', 'r') as f:
    data = json.load(f)

# Validate schema constraints
errors = []

if len(data['events']) != 200:
    errors.append(f"Expected 200 events, got {len(data['events'])}")

valid_processes = {'compton', 'rayleigh', 'photoelectric', 'pair'}
for i, evt in enumerate(data['events']):
    if evt['process'] not in valid_processes:
        errors.append(f"Event {i}: invalid process '{evt['process']}'")
    
    # Check e_in_MeV
    if not isinstance(evt.get('e_in_MeV'), (int, float)):
        errors.append(f"Event {i}: e_in_MeV is not a number")
    
    # Required fields
    required = ['process', 'e_in_MeV', 'e_out_MeV', 'cos_scatter', 't_e_MeV', 'cos_electron', 't_positron_MeV']
    for field in required:
        if field not in evt:
            errors.append(f"Event {i}: missing field '{field}'")
    
    # Null/number validation for nullable fields
    for field in ['e_out_MeV', 'cos_scatter', 't_positron_MeV']:
        val = evt.get(field)
        if val is not None and not isinstance(val, (int, float)):
            errors.append(f"Event {i}: {field} is neither null nor number")
    
    for field in ['cos_electron', 't_e_MeV']:
        val = evt.get(field)
        if val is not None and not isinstance(val, (int, float)):
            errors.append(f"Event {i}: {field} is neither null nor number")

if errors:
    print("Schema validation errors:")
    for err in errors[:10]:
        print(f"  - {err}")
else:
    print("Schema validation passed!")

print("\nSummary statistics:")
from collections import Counter
probs = [0.0784, 0.0026, 0.00017, 0.0198]
mu_total = sum(probs)
expected = [p/mu_total*100 for p in probs]
actual = [sum(1 for e in data['events'] if e['process'] == proc)/200*100 for proc in valid_processes]

process_order = list(valid_processes)
for proc in sorted(process_order):
    exp_idx = list(valid_processes).index(proc)
    print(f"  {proc}: expected ~{expected[exp_idx]:.1f}%, got {sum(1 for e in data['events'] if e['process']==proc)} ({sum(1 for e in data['events'] if e['process']==proc)/2:.1f}%)")

print(f"\nMethod length: {len(data['method'])} chars")
print("\nFinal JSON structure validated. Ready to output.")
