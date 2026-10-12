
sim = r'''
import numpy as np, sys, time
from phys import *
rng=np.random.default_rng(2024)
Vc=np.pi*1.0**2*0.2
# pyramid solid angle
ab=np.linspace(-0.05,0.05,801)
A,B=np.meshgrid(ab,ab)
Omega=np.trapezoid(np.trapezoid((1+A**2+B**2)**-1.5,ab,axis=1),ab)
Ecut=0.02
# Rayleigh angle table: pdf in theta
thg=np.linspace(0,np.pi,40001)
def rayl_pdf_theta(E):
    lam=0.012398/E; x=np.sin(thg/2)/lam
    return (1+np.cos(thg)**2)*np.exp(-(x/X0)**2)*np.sin(thg)
def rayl_sample(E,n):
    p=rayl_pdf_theta(E); c=np.concatenate([[0],np.cumsum(0.5*(p[1:]+p[:-1])*np.diff(thg))]); c/=c[-1]
    u=rng.random(n); th=np.interp(u,c,thg); return np.cos(th)
def rayl_q(E,cth):
    # dsigma/dOmega / sigma (per sr) with sigma normalized
    lam=0.012398/E; s=np.sqrt(np.clip((1-cth)/2,0,1)); x=s/lam
    f=(1+cth**2)*np.exp(-(x/X0)**2)
    return f/(2*np.pi*np.trapezoid(rayl_pdf_theta(E),thg))
def KN_eps(E,n):
    a=E/me; emin=1/(1+2*a); out=np.empty(0)
    res=[]; need=n
    while need>0:
        m=max(need*3,1000)
        eps=np.exp(rng.random(m)*np.log(emin))   # pdf ~ 1/eps
        cth=1-(1/eps-1)/a; s2=1-cth**2
        g=eps*(eps+1/eps-s2)/2.0
        acc=eps[rng.random(m)<g]
        res.append(acc); need-=len(acc)
    eps=np.concatenate(res)[:n]
    return eps
def KN_q(E,cth):
    a=E/me
    Ep_over_E=1/(1+a*(1-cth)); s2=1-cth**2
    dsig=0.5*re**2*Ep_over_E**2*(Ep_over_E+1/Ep_over_E-s2)
    return dsig/sig_KN(E)
def rotate(d,cth,phi):
    # d: (n,3); returns new directions
    n=len(d)
    ref=np.where(np.abs(d[:,2:3])<0.9, np.array([[0,0,1.0]]), np.array([[1.0,0,0]]))
    a=np.cross(d,ref); a/=np.linalg.norm(a,axis=1)[:,None]
    b=np.cross(d,a)
    sth=np.sqrt(np.clip(1-cth**2,0,1))
    return cth[:,None]*d+sth[:,None]*(np.cos(phi)[:,None]*a+np.sin(phi)[:,None]*b)
def isotropic(n):
    c=2*rng.random(n)-1; ph=2*np.pi*rng.random(n); s=np.sqrt(1-c*c)
    return np.stack([s*np.cos(ph),s*np.sin(ph),c],axis=1)
def sample_cyl(n):
    r=np.sqrt(rng.random(n)); ph=2*np.pi*rng.random(n)
    return np.stack([r*np.cos(ph), r*np.sin(ph), 4.9+0.2*rng.random(n)],axis=1)
def box_exit(pos,d):
    # distance to exit of box x,y in [-15,15], z in [0,30]
    with np.errstate(divide='ignore',invalid='ignore'):
        tx=np.where(d[:,0]>0,(15-pos[:,0])/d[:,0],np.where(d[:,0]<0,(-15-pos[:,0])/d[:,0],np.inf))
        ty=np.where(d[:,1]>0,(15-pos[:,1])/d[:,1],np.where(d[:,1]<0,(-15-pos[:,1])/d[:,1],np.inf))
        tz=np.where(d[:,2]>0,(30-pos[:,2])/d[:,2],np.where(d[:,2]<0,(0-pos[:,2])/d[:,2],np.inf))
    t=np.minimum(np.minimum(tx,ty),tz)
    return np.maximum(t,0)

def run(Ns, chunk=200000):
    cands=[]  # (type, weight, E_arr)
    # primary candidates
    done=0
    while done<Ns:
        n=min(chunk,Ns-done); done+=n
        # source photons
        Esrc=np.where(rng.random(n)<0.5,1.1732,1.3325)
        # sample a,b uniform in square with pdf ~ (1+a2+b2)^-1.5
        a=np.empty(0); bb=np.empty(0); 
        aa=[];bbs=[];
        got=0
        while got<n:
            m=2*(n-got)+100
            aa_=rng.uniform(-0.05,0.05,m); bb_=rng.uniform(-0.05,0.05,m)
            acc=rng.random(m)<(1+aa_**2+bb_**2)**-1.5
            aa.append(aa_[acc]); bbs.append(bb_[acc]); got+=acc.sum()
        a=np.concatenate(aa)[:n]; bb=np.concatenate(bbs)[:n]
        d=np.stack([a,bb,np.ones(n)],axis=1); d/=np.linalg.norm(d,axis=1)[:,None]
        src=np.array([0,0,-100.0])
        entry=src+d*(100/d[:,2])[:,None]
        # primary candidate: one uniform x in cylinder per source photon
        x=sample_cyl(n)
        vec=x-src; R=np.linalg.norm(vec,axis=1)
        Lw=np.linalg.norm(x-(src+vec/ (vec[:,2:3]/ (100+x[:,2:3]) ) *0)[:,:],axis=1)*0  # placeholder
        # path length in water from entry plane to x along ray from source
        t_in=100.0/(vec[:,2])            # fraction along vec at which z=0
        Lpath=(1-t_in)*R
        mu=mu_tot(Esrc)
        w=Vc*mu*np.exp(-mu*Lpath)/(Omega*R**2)
        cands.append(np.stack([np.zeros(n), w, Esrc],axis=1))
        # transport photons
        pos=entry.copy(); dr=d.copy(); E=Esrc.copy()
        while len(E)>0:
            mu=mu_tot(E)
            s=-np.log(rng.random(len(E)))/mu
            dbox=box_exit(pos,dr)
            inside=s<dbox
            pos=pos[inside]; dr=dr[inside]; E=E[inside]; s=s[inside]; mu=mu[inside]
            if len(E)==0: break
            y=pos+s[:,None]*dr
            # NEE candidates
            xs=sample_cyl(len(E))
            rv=xs-y; r=np.linalg.norm(rv,axis=1); u=rv/r[:,None]
            cth=np.sum(dr*u,axis=1)
            mc=mu_c(E); mR=mu_R(E); mP=mu_pair(E); mpe=mu_pe(E)
            # Compton NEE
            # E' from cos
            al=E/me
            Ep=E/(1+al*(1-cth))
            muEp=mu_tot(Ep)
            qc=KN_q(E,cth)
            tau=muEp*r
            wC=(mc/mu)*Vc*muEp*qc*np.exp(-tau)/r**2
            cands.append(np.stack([np.ones(len(E))*1, wC, Ep],axis=1))
            # Rayleigh NEE
            qR=rayl_q(E,cth)
            wR=(mR/mu)*Vc*mu*qR*np.exp(-mu*r)/r**2
            cands.append(np.stack([np.ones(len(E))*2, wR, E],axis=1))
            # pair NEE (two 511 photons, isotropic)
            m511=mu_tot(0.511)
            wP=(mP/mu)*Vc*2*m511*np.exp(-m511*r)/(4*np.pi*r**2)
            cands.append(np.stack([np.ones(len(E))*3, wP, np.full(len(E),0.511)],axis=1))
            # analog process
            pc=mc/mu; pR=mR/mu; pP=mP/mu
            uu=rng.random(len(E))
            isC=uu<pc
            isR=(uu>=pc)&(uu<pc+pR)
            isP=(uu>=pc+pR)&(uu<pc+pR+pP)
            newpos=[];newdir=[];newE=[]
            # Compton continuing
            if isC.any():
                Ei=E[isC]; ai=Ei/me
                eps=KN_eps(Ei.mean() if False else 1.0,0) if False else None
                # sample eps per photon with its own energy (vectorized by rejection with per-element a)
                k=len(Ei); epsv=np.empty(k); todo=np.arange(k)
                while len(todo)>0:
                    aa_=ai[todo]; emin=1/(1+2*aa_)
                    e=np.exp(rng.random(len(todo))*np.log(emin))
                    cc=1-(1/e-1)/aa_; s2=1-cc**2
                    g=e*(e+1/e-s2)/2.0
                    ok=rng.random(len(todo))<g
                    epsv[todo[ok]]=e[ok]; todo=todo[~ok]
                Eout=epsv*Ei
                cc=1-(1/epsv-1)/ai
                dC=rotate(dr[isC],cc,2*np.pi*rng.random(k))
                newpos.append(y[isC]); newdir.append(dC); newE.append(Eout)
            if isR.any():
                ci=rayl_sample(1.0,0) if False else None
                k=isR.sum(); Ei=E[isR]
                cR=np.empty(k)
                for Ev in np.unique(Ei):
                    msk=Ei==Ev
                    cR[msk]=rayl_sample(Ev,msk.sum())
                dR=rotate(dr[isR],cR,2*np.pi*rng.random(k))
                newpos.append(y[isR]); newdir.append(dR); newE.append(Ei)
            if isP.any():
                k=isP.sum()
                for _ in range(2):
                    newpos.append(y[isP]); newdir.append(isotropic(k)); newE.append(np.full(k,0.511))
            if newE:
                pos=np.concatenate(newpos); dr=np.concatenate(newdir); E=np.concatenate(newE)
                keep=E>Ecut
                pos=pos[keep]; dr=dr[keep]; E=E[keep]
            else:
                pos=np.zeros((0,3)); dr=np.zeros((0,3)); E=np.zeros(0)
    C=np.concatenate(cands)
    return C

if __name__=="__main__":
    t0=time.time()
    Ns=int(sys.argv[1])
    C=run(Ns)
    np.save('cands.npy',C)
    print('n cands',len(C),'time',time.time()-t0)
    for ty in [0,1,2,3]:
        m=C[:,0]==ty
        print(ty, m.sum(), C[m,1].sum()/Ns)
'''
open('sim.py','w').write(sim)
r=subprocess.run(['python','sim.py','20000'],capture_output=True,text=True,cwd='.')
print(r.stdout, r.stderr[-3000:])
