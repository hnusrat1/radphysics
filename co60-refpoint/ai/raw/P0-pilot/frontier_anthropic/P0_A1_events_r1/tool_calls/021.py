
import json
d=json.load(open('answer.json'))
def r4(x): return None if x is None else float(f"{x:.4g}")
ev2=[{k:(v if k=='process' else r4(v)) for k,v in e.items()} for e in d['events']]
s=",\n".join(json.dumps(e,separators=(',',':')) for e in ev2)
print(s[16900:28200])
