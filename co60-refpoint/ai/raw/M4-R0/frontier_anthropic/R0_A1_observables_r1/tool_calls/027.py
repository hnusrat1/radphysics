
import json
d=json.load(open('/tmp/work/results.json'))['res']
for k,v in d.items():
    if isinstance(v,list): print(k, [round(x,5) for x in v])
    elif isinstance(v,dict): print(k, {a:round(b,5) for a,b in v.items()})
    else: print(k, v)
