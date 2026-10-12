import json
import sys

try:
    with open('final.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    # Re-read original and process
    with open('full_output.json', 'r') as f:
        d = json.load(f)
    
    def round_val(v):
        if isinstance(v, float): return round(v, 2) if abs(v)<1.01 else round(v, 1)
        return v
    
    for ev in d['events']:
        ev['e_in_MeV'] = round(ev['e_in_MeV'], 1)
        if ev['e_out_MeV'] is not None: ev['e_out_MeV'] = round(ev['e_out_MeV'], 1)
        if ev['cos_scatter'] is not None: ev['cos_scatter'] = round(ev['cos_scatter'], 2)
        if ev['cos_electron'] is not None: ev['cos_electron'] = round(ev['cos_electron'], 2)
        if ev['t_e_MeV'] is not None: ev['t_e_MeV'] = round(ev['t_e_MeV'], 1)
        if ev['t_positron_MeV'] is not None: ev['t_positron_MeV'] = round(ev['t_positron_MeV'], 1)
        
    out_str = json.dumps(d, separators=(',', ':'))
    with open('final.json', 'w') as f:
        f.write(out_str)
    print("Generated and saved.")

# Now print
if True:
    with open('final.json', 'r') as f:
        s = f.read()
    print(s[:50] + "...")
    print(f"Total length: {len(s)}")