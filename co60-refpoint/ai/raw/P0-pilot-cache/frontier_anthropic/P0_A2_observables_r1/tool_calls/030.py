
import sys, time, pickle; sys.path.insert(0,'/tmp/work')
import numpy as np, mcph, mcph2
t=time.time()
tot=None; births=[]
NB=20; NP=1000000
for b in range(NB):
    r=mcph2.run(NP,100+b)
    births.append(r['births'])
    if tot is None:
        tot={k:(r[k] if not isinstance(r[k],dict) else {kk:vv.copy() for kk,vv in r[k].items()}) for k in ['flu_E','flu_c','en_fl','intr','etr','Qslab']}
    else:
        for k in ['flu_E','flu_c','en_fl','Qslab']: tot[k]=tot[k]+r[k]
        for k in ['intr','etr']:
            for kk in tot[k]: tot[k][kk]=tot[k][kk]+r[k][kk]
tot['births']=np.vstack(births); tot['N']=NB*NP
pickle.dump(tot,open('/tmp/work/tot.pkl','wb'))
print(time.time()-t, tot['births'].shape, tot['flu_E'].sum(axis=1)/tot['N']/(np.pi*0.2*mcph.PHI0_PER_SRC))
PHI=mcph.PHI0_PER_SRC
print(np.round(tot['Qslab'][:12]/(tot['N']*np.pi*9*0.25*PHI),5))
print(np.round(tot['Qslab'][[20,30,40,47]]/(tot['N']*np.pi*9*0.25*PHI),5))
