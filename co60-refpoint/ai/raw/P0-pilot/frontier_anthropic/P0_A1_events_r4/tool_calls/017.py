
import json
ev=json.load(open('/tmp/work/answer.json'))['events']
def f(x,nd=4): return 'null' if x is None else (repr(round(x,nd)))
lines=['{"process":"%s","e_in_MeV":%s,"e_out_MeV":%s,"cos_scatter":%s,"t_e_MeV":%s,"cos_electron":%s,"t_positron_MeV":%s}'%(e['process'],f(e['e_in_MeV']),f(e['e_out_MeV']),f(e['cos_scatter']),f(e['t_e_MeV']),f(e['cos_electron']),f(e['t_positron_MeV'])) for e in ev]
print("\n".join(lines[100:200]))
