
import numpy as np, sys, time, pickle, importlib
sys.path.insert(0,'/tmp/work')
import sim; importlib.reload(sim)
t=time.time()
allseg={}; hist=np.zeros(300)
NB=8; NPER=2000000
for b in range(NB):
    seg,dd=sim.run(NPER,100+b)
    for k,v in seg.items(): allseg.setdefault(k,[]).append(v)
    hist+=dd['hist_d']
seg={k:np.concatenate(v) for k,v in allseg.items()}
NTOT=NB*NPER
pickle.dump((seg,hist,NTOT),open('/tmp/work/tallies.pkl','wb'))
print(time.time()-t, NTOT, seg['tl'].size)
