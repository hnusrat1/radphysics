import json
import sys

with open('full_output.json', 'r') as f:
    data = json.load(f)

# Aggressive rounding to minimize size
def round_val(v):
    if isinstance(v, float):
        # For cosines, keep 2 decimals; others 1
        if abs(v) < 1.01 and v != 0.0: return round(v, 2) # Cosines ~1 or less
        return round(v, 1)
    return v

for ev in data['events']:
    # Fix e_in_MeV precision (1.2, 1.3)
    ev['e_in_MeV'] = round(ev['e_in_MeV'], 1)
    
    if ev['e_out_MeV'] is not None: ev['e_out_MeV'] = round(ev['e_out_MeV'], 1)
    if ev['cos_scatter'] is not None: ev['cos_scatter'] = round(ev['cos_scatter'], 2)
    if ev['cos_electron'] is not None: ev['cos_electron'] = round(ev['cos_electron'], 2)
    
    # Force t_e and tp to 1 decimal
    if ev['t_e_MeV'] is not None: ev['t_e_MeV'] = round(ev['t_e_MeV'], 1)
    if ev['t_positron_MeV'] is not None: ev['t_positron_MeV'] = round(ev['t_positron_MeV'], 1)

out_str = json.dumps(data, separators=(',', ':'))
sys.stdout.write(out_str)
sys.stdout.flush()