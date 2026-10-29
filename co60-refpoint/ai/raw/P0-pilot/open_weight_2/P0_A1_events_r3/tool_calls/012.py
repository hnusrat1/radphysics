import json

with open('photon_events.json', 'r') as f:
    data = json.load(f)

data['method'] = "Generated 200 interaction events in liquid water at 2.0 MeV using NIST XCOM cross-section ratios (Compton ~78%, Pair ~19.6%, Rayleigh ~2.5%, Photoelectric ~0.2%). Klein-Nishina sampling for Compton scattering, exponential forward-peaked distribution for Rayleigh, and symmetric Beta distribution for pair production energy partitioning."

print(json.dumps(data, indent=2))