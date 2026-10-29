
import json

# Read and verify the JSON file
with open('events.json', 'r') as f:
    data = json.load(f)

print(f"Number of events: {len(data['events'])}")
print(f"Method text:\n{data['method']}\n")

# Verify event structure
from collections import Counter
process_counts = Counter(ev['process'] for ev in data['events'])
print(f"Process distribution: {dict(process_counts)}")

# Check energy conservation for Compton events
compton_events = [ev for ev in data['events'] if ev['process'] == 'compton']
energy_conservation_ok = True
for ev in compton_events[:5]:  # Check first 5
    if abs(ev['e_in_MeV'] - (ev['e_out_MeV'] + ev['t_e_MeV'])) > 1e-6:
        print(f"Energy conservation failed: E_in={ev['e_in_MeV']}, E_out+t_e={ev['e_out_MeV']+ev['t_e_MeV']}")
        energy_conservation_ok = False

if energy_conservation_ok:
    print("Energy conservation OK for sampled Compton events")

# Check pair production energy sharing
pair_events = [ev for ev in data['events'] if ev['process'] == 'pair']
pair_energy_check = all(
    abs((ev['t_e_MeV'] + ev['t_positron_MeV']) - (2.0 - 2*0.510998950)) < 0.01
    for ev in pair_events
)
print(f"Pair production energy sharing OK: {pair_energy_check}")
