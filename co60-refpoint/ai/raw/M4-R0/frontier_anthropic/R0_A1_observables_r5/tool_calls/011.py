
import numpy as np, sys, time, pickle
sys.path.insert(0,'/tmp/work')
import sim
t=time.time()
allseg={}; alldep={}
NB=8; NPER=2000000
for b in range(NB):
    seg,dep=sim.run(NPER,100+b)
    for k,v in seg.items(): allseg.setdefault(k,[]).append(v)
    for k,v in dep.items(): alldep.setdefault(k,[]).append(v)
seg={k:np.concatenate(v) for k,v in allseg.items()}
dep={k:np.concatenate(v) for k,v in alldep.items()}
NTOT=NB*NPER
pickle.dump((seg,dep,NTOT),open('/tmp/work/tallies.pkl','wb'))
print(time.time()-t, NTOT, seg['tl'].size, dep['z'].size)
