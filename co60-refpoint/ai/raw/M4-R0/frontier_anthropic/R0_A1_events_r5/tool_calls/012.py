
import json
d=json.load(open("sample.json"))
keys=["process","e_in_MeV","e_out_MeV","cos_scatter","t_e_MeV","cos_electron","t_positron_MeV"]
def fmt(x):
    if x is None: return "null"
    if isinstance(x,str): return '"'+x+'"'
    return f"{x:.4f}" if abs(x)>=1e-3 or x==0 else f"{x:.2e}"
lines=[]
for e in d:
    lines.append("["+",".join(fmt(e[k]) for k in keys)+"]")
print(len(lines))
print("\n".join(lines[:100]))
