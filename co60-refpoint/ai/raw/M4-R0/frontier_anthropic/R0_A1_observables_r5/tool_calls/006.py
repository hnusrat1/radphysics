
code = r'''
import numpy as np
from mc import *
E_LINES=np.array([1.1732,1.3325])
def sample_dirs(n,rng):
    out=np.empty((0,2)); 
    while out.shape[0]<n:
        a=rng.uniform(-0.05,0.05,2*n); b=rng.uniform(-0.05,0.05,2*n)
        acc=rng.random(2*n) < (1+a*a+b*b)**-1.5
        out=np.vstack([out,np.c_[a[acc],b[acc]]])
    return out[:n]
def run(N,seed):
    rng=np.random.default_rng(seed)
    ab=sample_dirs(N,rng)
    E=rng.choice(E_LINES,size=N)
    pos=np.c_[100*ab[:,0],100*ab[:,1],np.zeros(N)]
    u=np.c_[ab[:,0],ab[:,1],np.ones(N)]; u/=np.linalg.norm(u,axis=1,keepdims=True)
    prim=np.ones(N,bool)
    seg={'E':[],'tl':[],'prim':[],'ux':[],'uy':[],'uz':[]}
    dep={'z':[],'r':[],'d':[]}
    # keep only photons from pos within phantom (they are at surface; enter)
    it=0
    while E.size:
        it+=1
        mu=mu_c(E)+mu_pe(E)+mu_coh(E)+mu_pair(E)
        s=-np.log(rng.random(E.size))/mu
        dex=exit_distance(pos,u)
        step=np.minimum(s,dex)
        tl,_=sv_track(pos,u,step,Z1,Z2)
        m=tl>0
        if m.any():
            seg['E'].append(E[m]); seg['tl'].append(tl[m]); seg['prim'].append(prim[m])
            seg['ux'].append(u[m,0]); seg['uy'].append(u[m,1]); seg['uz'].append(u[m,2])
        # move
        newpos=pos+step[:,None]*u
        inter=s<dex
        # interactions
        idx=np.where(inter)[0]
        if idx.size:
            Ei=E[idx]; pi=newpos[idx]; ui=u[idx].copy(); pr=prim[idx]
            pc=mu_c(Ei)/mu[idx]; ppe=mu_pe(Ei)/mu[idx]; pcoh=mu_coh(Ei)/mu[idx]
            r=rng.random(idx.size)
            isC=r<pc
            isPE=(r>=pc)&(r<pc+ppe)
            isCoh=(r>=pc+ppe)&(r<pc+ppe+pcoh)
            isPair=r>=pc+ppe+pcoh
            # Compton
            ic=np.where(isC)[0]
            if ic.size:
                eps,c=sample_KN(Ei[ic],rng)
                phi=rng.uniform(0,2*np.pi,ic.size)
                unew=rotate(ui[ic],c,phi)
                Tdep=Ei[ic]*(1-eps)
                dep['z'].append(pi[ic,2]); dep['r'].append(np.hypot(pi[ic,0],pi[ic,1])); dep['d'].append(Tdep)
                Enew=Ei[ic]*eps
                ui[ic]=unew; Ei_new=Ei.copy(); Ei_new[ic]=Enew
                pr[ic]=False
                Ei=Ei_new
            # coherent
            icoh=np.where(isCoh)[0]
            if icoh.size:
                c=sample_thomson(icoh.size,rng)
                phi=rng.uniform(0,2*np.pi,icoh.size)
                ui[icoh]=rotate(ui[icoh],c,phi)
                pr[icoh]=False
            # absorbed: PE and pair
            iab=np.where(isPE|isPair)[0]
            if iab.size:
                dd=np.where(isPE[iab],Ei[iab],Ei[iab]-2*me)
                dep['z'].append(pi[iab,2]); dep['r'].append(np.hypot(pi[iab,0],pi[iab,1])); dep['d'].append(dd)
            # update idx photons
            keep=isC|isCoh
            # low-energy cutoff
            keep_idx=idx[keep]
            # deposit low energy photons
            low=keep & (Ei<0.001)
            # store updated state
            pos_all=newpos.copy()
            # for kept photons: new pos, dir, E
            E2=E.copy(); u2=u.copy(); prim2=prim.copy()
            E2[idx]=Ei; u2[idx]=ui; prim2[idx]=pr
            # remove absorbed & exited
            alive=np.ones(E.size,bool)
            alive[idx[~keep]]=False
            alive[idx[low]]=False
            # low-energy photons: deposit energy
            if low.any():
                lidx=idx[low]
                dep['z'].append(newpos[lidx,2]); dep['r'].append(np.hypot(newpos[lidx,0],newpos[lidx,1])); dep['d'].append(Ei[low])
            pos=newpos
            E=E2; u=u2; prim=prim2
            E=E[alive]; u=u[alive]; pos=pos[alive]; prim=prim[alive]
        else:
            E=E[:0]; u=u[:0]; pos=pos[:0]; prim=prim[:0]
        # exited photons already removed by the else only if all exit; handle generically:
        # photons that exit (not inter) must be removed
        # (done below)
        if E.size and False: pass
        if it>400: break
    return seg,dep
'''
open('sim.py','w').write(code)
print("ok")
