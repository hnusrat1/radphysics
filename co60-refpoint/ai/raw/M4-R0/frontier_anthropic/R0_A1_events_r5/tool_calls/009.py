
sample_code = r'''
import pickle, json, numpy as np
from collections import Counter
ev = pickle.load(open("events_all.pkl","rb"))
rng = np.random.default_rng(7)
idx = rng.choice(len(ev), size=200, replace=False)
out=[]
for i in idx:
    e = ev[i]
    d = {}
    d['process']=e['process']
    d['e_in_MeV']=round(e['e_in_MeV'],5)
    d['e_out_MeV']=None if e['e_out_MeV'] is None else round(e['e_out_MeV'],5)
    d['cos_scatter']=None if e['cos_scatter'] is None else round(e['cos_scatter'],6)
    d['t_e_MeV']=round(e['t_e_MeV'],5)
    d['cos_electron']=None if e['cos_electron'] is None else round(e['cos_electron'],6)
    d['t_positron_MeV']=None if e['t_positron_MeV'] is None else round(e['t_positron_MeV'],5)
    out.append(d)
print(Counter(o['process'] for o in out))
print(Counter(round(o['e_in_MeV'],4) for o in out).most_common(5))
json.dump(out, open("sample.json","w"))
print(json.dumps(out[:3]))
'''
open('sample.py','w').write(sample_code)
r=subprocess.run(['python','sample.py'],capture_output=True,text=True); print(r.stdout, r.stderr[-2000:])
