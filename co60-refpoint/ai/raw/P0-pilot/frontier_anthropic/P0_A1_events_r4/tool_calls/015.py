
import json
ev=json.load(open('/tmp/work/answer.json'))['events']
def f(x,nd=4): return 'null' if x is None else (repr(round(x,nd)))
lines=[]
for e in ev:
    lines.append('{"process":"%s","e_in_MeV":%s,"e_out_MeV":%s,"cos_scatter":%s,"t_e_MeV":%s,"cos_electron":%s,"t_positron_MeV":%s}'%(e['process'],f(e['e_in_MeV'],4),f(e['e_out_MeV'],4),f(e['cos_scatter'],4),f(e['t_e_MeV'],4),f(e['cos_electron'],4),f(e['t_positron_MeV'],4)))
print(len(lines), sum(len(l) for l in lines))
print("\n".join(lines[:100]))
