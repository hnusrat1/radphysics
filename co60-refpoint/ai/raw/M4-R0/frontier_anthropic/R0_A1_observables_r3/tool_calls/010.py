
import sys, time, pickle; sys.path.insert(0,'/tmp/work')
import mc as M, physics as P
import numpy as np
zpts=[0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0]
t=time.time()
R=M.run(1000000, zpts, seed=21, store_main=zpts.index(5.0))
print("time",time.time()-t, "collisions",R['nColl'])
N=R['N']
for j,z in enumerate(zpts):
    print(z, "scat fluence/Phi0=", P.SF/N*R['A_tot'][j], "K_col_scat=", P.SF/N*R['A_Kc'][j], "K_scat=",P.SF/N*R['A_K'][j])
# save
st=R['store']
Es=np.concatenate([s[0] for s in st]); rh=np.concatenate([s[1] for s in st]); W=np.concatenate([s[2] for s in st]); tg=np.concatenate([s[3] for s in st])
np.savez('/tmp/work/main5.npz',E=Es,rh=rh,W=W,tag=tg)
pickle.dump({k:v for k,v in R.items() if k!='store'},open('/tmp/work/R.pkl','wb'))
print(len(Es))
