
import sys, time, pickle; sys.path.insert(0,'/tmp/work')
import mc2 as M2, physics as P, numpy as np
zpts=[0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0]
im=zpts.index(5.0)
chunks=[]
t=time.time()
for s in range(4):
    chunks.append(M2.run(2500000, zpts, seed=200+s, store_main=im))
print("time",time.time()-t)
keys=['A_E','A_c','A_tot','A_Ef','A_K','A_Kc','A_int','A_tr','A_Kc_scat']
tot={k:sum(c[k] for c in chunks) for k in keys}
Ntot=sum(c['N'] for c in chunks)
store=[]
for c in chunks: store+=c['store']
Es=np.concatenate([s[0] for s in store]); ds=np.concatenate([s[1] for s in store]); Ws=np.concatenate([s[2] for s in store])
np.savez('/tmp/work/main5_seg.npz',E=Es,d=ds,W=Ws)
pickle.dump(dict(tot=tot,N=Ntot,zpts=zpts),open('/tmp/work/tot.pkl','wb'))
ratios=np.array([[P.SF/c['N']*c['A_tot'][j] for j in range(len(zpts))] for c in chunks])
print("scat at 5cm chunks:", np.round(ratios[:,im],4))
Kc_scat=P.SF/Ntot*tot['A_Kc_scat']; Ks=P.SF/Ntot*tot['A_K']
for j,z in enumerate(zpts):
    prim=0.5*sum(np.exp(-P.mutot(E)*z)*P.KC(E) for E in P.lines)
    primK=0.5*sum(np.exp(-P.mutot(E)*z)*P.ET(E) for E in P.lines)
    print(z, round(P.SF/Ntot*tot['A_tot'][j],4), "Kcol_tot", round(prim+Kc_scat[j],5), "K_tot", round(primK+Ks[j],5), " dose-like Kcol scat share", round(Kc_scat[j]/(prim+Kc_scat[j]),4))
