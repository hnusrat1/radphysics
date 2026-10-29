
mc = r'''
import numpy as np, sys, time, json
sys.path.insert(0,'/tmp/work')
from xs import mu_of, mu_inc, mu_pair, mu_pe, mu_coh, mu_tot, sKN, ne
MEc=0.51099895
LO=np.array([-15.,-15.,0.]); HI=np.array([15.,15.,30.])

def rand_dirs_perp(d, rng):
    n=d.shape[0]
    a=np.zeros_like(d); sel=np.abs(d[:,0])<0.9
    a[sel,0]=1.0; a[~sel,1]=1.0
    e1=np.cross(a,d); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(d,e1)
    phi=rng.uniform(0,2*np.pi,n)
    return np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2

def sample_compton(E, rng):
    al=E/MEc; emin=1/(1+2*al)
    eps=np.empty_like(E); cth=np.empty_like(E)
    todo=np.ones(E.shape[0],bool)
    while todo.any():
        idx=np.where(todo)[0]
        r=rng.uniform(size=idx.size)
        e=emin[idx]**r
        a=al[idx]
        c=1-(1/e-1)/a
        s2=1-c*c
        acc=rng.uniform(size=idx.size)<(1+e*e-e*s2)/2
        ok=idx[acc]
        eps[ok]=e[acc]; cth[ok]=c[acc]
        todo[ok]=False
    return eps,cth

def sample_photo_cos(E, rng):
    beta=np.sqrt(1-(MEc/(E+MEc))**2)
    out=np.empty_like(E); todo=np.ones(E.shape[0],bool)
    M=1/(1-beta)**4
    while todo.any():
        idx=np.where(todo)[0]
        c=rng.uniform(-1,1,idx.size)
        f=(1-c*c)/(1-beta[idx]*c)**4
        acc=rng.uniform(size=idx.size)*M[idx]<f
        ok=idx[acc]; out[ok]=c[acc]; todo[ok]=False
    return out

def run_chunk(N, rng, events):
    # source photons
    tx=rng.uniform(-0.05,0.05,N); ty=rng.uniform(-0.05,0.05,N)
    acc=rng.uniform(size=N)<(1+tx*tx+ty*ty)**(-1.5)
    tx=tx[acc]; ty=ty[acc]; n=np.sqrt(tx*tx+ty*ty+1)
    d=np.stack([tx/n,ty/n,1/n],1)
    p=np.stack([100*tx,100*ty,np.zeros_like(tx)],1)
    E=np.full(p.shape[0],2.0)
    nsrc=p.shape[0]
    it=0
    while E.size>0:
        it+=1
        mu=mu_of(E)
        s=-np.log(rng.uniform(size=E.size))/mu
        with np.errstate(divide='ignore',invalid='ignore'):
            tb=np.where(d>0,(HI-p)/d,np.where(d<0,(LO-p)/d,np.inf))
        texit=tb.min(axis=1)
        inter=s<texit
        p=p[inter]; d=d[inter]; E=E[inter]; s=s[inter]; mu=mu[inter]
        if E.size==0: break
        E=E.copy()
        q=p+s[:,None]*d
        # kill very low-energy photons
        keep=E>=0.01
        q=q[keep]; d=d[keep]; E=E[keep]; mu=mu[keep]
        if E.size==0: break
        inV=(q[:,0]**2+q[:,1]**2<=1.0)&(q[:,2]>=9.9)&(q[:,2]<=10.1)
        u=rng.uniform(size=E.size)*mu
        pinc=mu_inc(E); ppe=mu_pe(E); pcoh=mu_coh(E); ppair=mu_pair(E)
        is_c=u<pinc
        is_pe=(~is_c)&(u<pinc+ppe)
        is_coh=(~is_c)&(~is_pe)&(u<pinc+ppe+pcoh)
        is_pair=(~is_c)&(~is_pe)&(~is_coh)
        newp=[];newd=[];newE=[]
        # Compton
        ic=np.where(is_c)[0]
        if ic.size:
            eps,cth=sample_compton(E[ic],rng)
            Ein=E[ic]; Eout=eps*Ein; Te=Ein-Eout
            sth=np.sqrt(np.clip(1-cth*cth,0,None))
            dn=d[ic]
            perp=rand_dirs_perp(dn,rng)
            nd=cth[:,None]*dn+sth[:,None]*perp
            ce=(Ein-Eout*cth)/np.sqrt(Ein**2+Eout**2-2*Ein*Eout*cth)
            ce=np.clip(ce,-1,1)
            m=inV[ic]
            for k in np.where(m)[0]:
                events.append(("compton",float(Ein[k]),float(Eout[k]),float(cth[k]),float(Te[k]),float(ce[k]),None))
            newp.append(q[ic]); newd.append(nd); newE.append(Eout)
        # Photoelectric
        ip=np.where(is_pe)[0]
        if ip.size:
            cpe=sample_photo_cos(E[ip],rng)
            m=inV[ip]
            for k in np.where(m)[0]:
                events.append(("photoelectric",float(E[ip][k]),None,None,float(E[ip][k]),float(cpe[k]),None))
        # Coherent
        ih=np.where(is_coh)[0]
        if ih.size:
            lam=0.012398/E[ih]
            sig=0.6*lam
            th=sig*np.sqrt(-2*np.log(rng.uniform(size=ih.size)))
            cth=np.cos(th); sth=np.sin(th)
            dn=d[ih]; perp=rand_dirs_perp(dn,rng)
            nd=cth[:,None]*dn+sth[:,None]*perp
            m=inV[ih]
            for k in np.where(m)[0]:
                events.append(("rayleigh",float(E[ih][k]),float(E[ih][k]),float(cth[k]),0.0,None,None))
            newp.append(q[ih]); newd.append(nd); newE.append(E[ih])
        # Pair
        ipr=np.where(is_pair)[0]
        if ipr.size:
            T=E[ipr]-2*MEc
            f=rng.uniform(size=ipr.size)
            Te=f*T; Tp=(1-f)*T
            th=np.abs(rng.normal(size=ipr.size))*MEc/E[ipr]
            ce=np.cos(th)
            m=inV[ipr]
            for k in np.where(m)[0]:
                events.append(("pair",float(E[ipr][k]),None,None,float(Te[k]),float(ce[k]),float(Tp[k])))
            # annihilation photons (two back-to-back 511 keV), isotropic
            z=rng.normal(size=(ipr.size,3)); z/=np.linalg.norm(z,axis=1)[:,None]
            newp.append(q[ipr]); newd.append(z); newE.append(np.full(ipr.size,MEc))
            newp.append(q[ipr]); newd.append(-z); newE.append(np.full(ipr.size,MEc))
        if newE:
            p=np.concatenate(newp); d=np.concatenate(newd); E=np.concatenate(newE)
        else:
            p=np.zeros((0,3)); d=np.zeros((0,3)); E=np.zeros(0)
    return

if __name__=="__main__":
    Ntot=int(sys.argv[1]); seed=int(sys.argv[2]); chunk=int(sys.argv[3])
    rng=np.random.default_rng(seed)
    events=[]
    t0=time.time()
    done=0
    while done<Ntot:
        n=min(chunk,Ntot-done)
        run_chunk(n,rng,events)
        done+=n
    print("histories",Ntot,"events",len(events),"time",time.time()-t0)
    import pickle
    with open(f"/tmp/work/events_{seed}.pkl","wb") as f: pickle.dump(events,f)
    from collections import Counter
    print(Counter(e[0] for e in events))
'''
with open('/tmp/work/mc.py','w') as f: f.write(mc)
import subprocess
r=subprocess.run(['python3','/tmp/work/mc.py','1000000','1','250000'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout, r.stderr[-3000:])
