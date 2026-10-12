import json

# Print first 10 events
print("First 10 events:")
for i, e in enumerate(ev_final[:10]):
    print(f"  {i}: process={e['process']}, e_in={e['e_in_MeV']:.4f}, e_out={e['e_out_MeV']}, cos_scat={e['cos_scatter']}, t_e={e['t_e_MeV']}, cos_e={e['cos_electron']}, t_pos={e['t_positron_MeV']}")

# Summary stats
import statistics
e_in_vals = [e['e_in_MeV'] for e in ev_final if e['e_in_MeV'] is not None]
e_out_vals = [e['e_out_MeV'] for e in ev_final if e['e_out_MeV'] is not None]
t_e_vals = [e['t_e_MeV'] for e in ev_final if e['t_e_MeV'] is not None and e['t_e_MeV']>0]
print(f"\nSummary:")
print(f"  e_in: min={min(e_in_vals):.4f}, max={max(e_in_vals):.4f}, mean={sum(e_in_vals)/len(e_in_vals):.4f}")
if e_out_vals:
    print(f"  e_out: min={min(e_out_vals):.4f}, max={max(e_out_vals):.4f}, mean={sum(e_out_vals)/len(e_out_vals):.4f}")
if t_e_vals:
    print(f"  t_e: min={min(t_e_vals):.4f}, max={max(t_e_vals):.4f}, mean={sum(t_e_vals)/len(t_e_vals):.4f}")

# Check if any scattered photons
scattered_count = sum(1 for e in ev_final if e['e_in_MeV'] < 1.0)
print(f"  Events with incident energy < 1.0 MeV (scattered): {scattered_count}")

# Distribution of incident energies
from collections import Counter
energy_count = Counter([round(e['e_in_MeV'], 4) for e in ev_final])
print(f"  Energy distribution (rounded to 4 decimals):")
for k in sorted(energy_count.keys()):
    print(f"    {k:.4f} MeV: {energy_count[k]}")