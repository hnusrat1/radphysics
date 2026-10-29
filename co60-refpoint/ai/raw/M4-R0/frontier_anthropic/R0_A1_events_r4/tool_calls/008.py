
import numpy as np, json
d=np.load('sample_99.npz')
names=['compton','rayleigh','photoelectric','pair']
ev=[]
N=len(d['proc'])
print(N, np.bincount(d['proc'],minlength=4), 'primary frac', d['prim'].mean())
rows=[]
for i in range(N):
    p=names[int(d['proc'][i])]
    e_in=float(d['Ein'][i])
    eo=d['Eout'][i]; cs=d['cosS'][i]; te=d['te'][i]; ce=d['cose'][i]; tp=d['tpos'][i]
    rows.append(dict(process=p,e_in=round(e_in,6),
        e_out=None if np.isnan(eo) else round(float(eo),6),
        cos_s=None if np.isnan(cs) else round(float(cs),5),
        te=round(float(te),6),
        ce=None if np.isnan(ce) else round(float(ce),5),
        tp=None if np.isnan(tp) else round(float(tp),6), prim=bool(d['prim'][i])))
print(rows[:5])
print(np.mean([r['prim'] for r in rows]))
from collections import Counter
print(Counter(r['e_in'] for r in rows))
