import json
import sys

with open('full_output.json', 'r') as f:
    data = json.load(f)

# Aggressively reduce precision to minimize size
def round_val(v, prec=3):
    if isinstance(v, float): return round(v, prec)
    return v

for ev in data['events']:
    for k, v in ev.items():
        ev[k] = round_val(v)

# Custom serializer to remove spaces inside dicts if possible? 
# json.dumps with separators does this.
out_str = json.dumps(data, separators=(',', ':'))
sys.stdout.write(out_str)
sys.stdout.flush()