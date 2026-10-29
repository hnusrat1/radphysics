
code = r'''
import numpy as np
from mc import *
E_LINES=np.array([1.1732,1.3325])
def sample_dirs(n,rng):
    out=np.empty((0,2))
    while out.shape[0]<n:
        a=rng.uniform(-0.05,0.05,2*n); b=rng.uniform(-0.05,0.05,2*n)
        acc=rng.random(2*n) < (1+a*a+b*b)**-1.5
        out=np.vstack([out,np.c_[a[acc],b[acc]]])
    return out[:n]
def run(N,seed,zmax_dep=12.0,rdep=3.0):
    rng=np.random.default_rng(seed)
    ab=sample_dirs(N,rng)
    E=rng.choice(E_LINES,size=N)
    pos=np.c_[100*ab[:,0],100*ab[:,1],np.zeros(N)]
    u=np.c_[ab[:,0],ab[:,1],np.ones(N)]; u/=np.linalg.norm(u,axis=1,keepdims=True)
    prim=np.ones(N,bool)
    seg={'E':[],'tl':[],'prim':[],'ux':[],'uy':[],'uz':[]}
    dep={'z':[],'r':[],'d':[],'E':[]}
    it=0
    while E.size:
        it+=1
        mu=mu_c(E)+mu_pe(E)+mu_coh(E)+mu_pair(E)
        s=-np.log(rng.random(E.size))/mu
        dex=exit_distance(pos,u)
        inter=s<dex
        step=np.where(inter,s,dex)
        tl,_=sv_track(pos,u,step,Z1,Z2)
        m=tl>0
        if m.any():
            seg['E'].append(E[m]); seg['tl'].append(tl[m]); seg['prim'].append(prim[m])
            seg['ux'].append(u[m,0]); seg['uy'].append(u[m,1]); seg['uz'].append(u[m,2])
        newpos=pos+step[:,None]*u
        # photons that exit: remove
        keep=inter.copy()
        # interacting photons
        idx=np.where(inter)[0]
        if idx.size:
            Ei=E[idx]; pi=newpos[idx]; ui=u[idx].copy(); pr=prim[idx].copy()
            mut=mu[idx]
            pc=mu_c(Ei)/mut; ppe=mu_pe(Ei)/mut; pcoh=mu_coh(Ei)/mut
            r=rng.random(idx.size)
            isC=r<pc
            isPE=(r>=pc)&(r<pc+ppe)
            isCoh=(r>=pc+ppe)&(r<pc+ppe+pcoh)
            isPair=r>=pc+ppe+pcoh
            alive_i=np.zeros(idx.size,bool)
            ic=np.where(isC)[0]
            if ic.size:
                eps,c=sample_KN(Ei[ic],rng)
                phi=rng.uniform(0,2*np.pi,ic.size)
                ui[ic]=rotate(ui[ic],c,phi)
                dep['z'].append(pi[ic,2]); dep['r'].append(np.hypot(pi[ic,0],pi[ic,1]))
                dep['d'].append(Ei[ic]*(1-eps)); dep['E'].append(Ei[ic])
                Ei[ic]=Ei[ic]*eps
                pr[ic]=False
                alive_i[ic]=True
            icoh=np.where(isCoh)[0]
            if icoh.size:
                c=sample_thomson(icoh.size,rng)
                phi=rng.uniform(0,2*np.pi,icoh.size)
                ui[icoh]=rotate(ui[icoh],c,phi)
                pr[icoh]=False
                alive_i[icoh]=True
            iab=np.where(isPE|isPair)[0]
            if iab.size:
                dd=np.where(isPE[iab],Ei[iab],Ei[iab]-2*me)
                dep['z'].append(pi[iab,2]); dep['r'].append(np.hypot(pi[iab,0],pi[iab,1]))
                dep['d'].append(dd); dep['E'].append(Ei[iab])
            # low-energy cutoff
            low=alive_i & (Ei<0.001)
            if low.any():
                dep['z'].append(pi[low,2]); dep['r'].append(np.hypot(pi[low,0],pi[low,1]))
                dep['d'].append(Ei[low]); dep['E'].append(Ei[low])
                alive_i[low]=False
            # write back
            newpos[idx]=pi
            E[idx]=Ei; u[idx]=ui; prim[idx]=pr
            keep[idx]=alive_i
        keep &= True
        pos=newpos[keep]; E=E[keep]; u=u[keep]; prim=prim[keep]
        if it>500: break
    out={k:np.concatenate(v) if len(v) else np.array([]) for k,v in seg.items()}
    dd={k:np.concatenate(v) if len(v) else np.array([]) for k,v in dep.items()}
    return out,dd
'''
open('sim.py','w').write(code)
import sim, time
t=time.time()
seg,dep=sim.run(20000,1)
print(time.time()-t, seg['tl'].size, dep['z'].size)
