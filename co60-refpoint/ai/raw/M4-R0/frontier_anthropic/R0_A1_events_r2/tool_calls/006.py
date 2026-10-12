
phys = r'''
import numpy as np
me=0.51099895; re=2.8179403e-13; NA=6.02214076e23
ne=NA*10/18.015
nmol=NA/18.015
X0=0.4
def sig_KN(E):
    E=np.asarray(E,float); a=E/me; t=np.log(1+2*a)
    return 2*np.pi*re**2*((1+a)/a**2*(2*(1+a)/(1+2*a)-t/a)+t/(2*a)-(1+3*a)/(1+2*a)**2)
def _sig_R_scalar(E):
    th=np.linspace(0,np.pi,20001); a=0.012398/E*X0
    s=np.sin(th/2); c=np.cos(th)
    f=0.5*re**2*(1+c**2)*100*np.exp(-(s/a)**2)*2*np.pi*np.sin(th)
    return np.trapezoid(f,th)
_Eg=np.logspace(np.log10(0.01),np.log10(2.0),300)
_sg=np.array([_sig_R_scalar(e) for e in _Eg])
def sig_R(E):
    E=np.asarray(E,float)
    return np.exp(np.interp(np.log(E),np.log(_Eg),np.log(_sg)))
def mu_c(E): return ne*sig_KN(E)
def mu_R(E): return nmol*sig_R(E)
def mu_pe(E):
    E=np.asarray(E,float); return 0.0025*(0.1/E)**3
def mu_pair(E):
    E=np.asarray(E,float); return np.where(E>1.022,0.0215*np.clip(1-1.022/E,0,None)**2.5,0.0)
def mu_tot(E): return mu_c(E)+mu_R(E)+mu_pe(E)+mu_pair(E)
'''
open('phys.py','w').write(phys)

sim = r'''
import numpy as np, sys, time
from phys import *
rng=np.random.default_rng(2024)
Vc=np.pi*1.0**2*0.2
ab=np.linspace(-0.05,0.05,801)
A,B=np.meshgrid(ab,ab)
Omega=np.trapezoid(np.trapezoid((1+A**2+B**2)**-1.5,ab,axis=1),ab)
Ecut=0.02
def rayl_q(E,cth):
    s2=np.clip((1-cth)/2,0,1); a=0.012398/E*X0
    f=0.5*re**2*(1+cth**2)*100*np.exp(-s2/a**2)
    return f/sig_R(E)
def rayl_sample(E):
    # proposal pdf in s ~ s exp(-(s/a)^2); accept with (1+c^2)/2
    n=len(E); a=0.012398/E*X0
    out=np.empty(n); todo=np.arange(n)
    while len(todo)>0:
        s=a[todo]*np.sqrt(-np.log(rng.random(len(todo))))
        s=np.minimum(s,1.0)
        c=1-2*s**2
        ok=rng.random(len(todo))<(1+c**2)/2
        out[todo[ok]]=c[ok]; todo=todo[~ok]
    return out
def KN_q(E,cth):
    a=E/me
    Pr=1/(1+a*(1-cth)); s2=1-cth**2
    dsig=0.5*re**2*Pr**2*(Pr+1/Pr-s2)
    return dsig/sig_KN(E)
def rotate(d,cth,phi):
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
    with np.errstate(divide='ignore',invalid='ignore'):
        tx=np.where(d[:,0]>0,(15-pos[:,0])/d[:,0],np.where(d[:,0]<0,(-15-pos[:,0])/d[:,0],np.inf))
        ty=np.where(d[:,1]>0,(15-pos[:,1])/d[:,1],np.where(d[:,1]<0,(-15-pos[:,1])/d[:,1],np.inf))
        tz=np.where(d[:,2]>0,(30-pos[:,2])/d[:,2],np.where(d[:,2]<0,(0-pos[:,2])/d[:,2],np.inf))
    return np.maximum(np.minimum(np.minimum(tx,ty),tz),0)
def KN_eps(Ei):
    k=len(Ei); ai=Ei/me; epsv=np.empty(k); todo=np.arange(k)
    while len(todo)>0:
        aa=ai[todo]; emin=1/(1+2*aa)
        e=np.exp(rng.random(len(todo))*np.log(emin))
        cc=1-(1/e-1)/aa; s2=1-cc**2
        g=e*(e+1/e-s2)/2.0
        ok=rng.random(len(todo))<g
        epsv[todo[ok]]=e[ok]; todo=todo[~ok]
    return epsv

def run(Ns, chunk=100000):
    cands=[]
    done=0
    while done<Ns:
        n=min(chunk,Ns-done); done+=n
        Esrc=np.where(rng.random(n)<0.5,1.1732,1.3325)
        aa=[];bbs=[];got=0
        while got<n:
            m=2*(n-got)+100
            a_=rng.uniform(-0.05,0.05,m); b_=rng.uniform(-0.05,0.05,m)
            acc=rng.random(m)<(1+a_**2+b_**2)**-1.5
            aa.append(a_[acc]); bbs.append(b_[acc]); got+=acc.sum()
        a=np.concatenate(aa)[:n]; bb=np.concatenate(bbs)[:n]
        d=np.stack([a,bb,np.ones(n)],axis=1); d/=np.linalg.norm(d,axis=1)[:,None]
        src=np.array([0,0,-100.0])
        entry=src+d*(100/d[:,2])[:,None]
        # primary candidate: one uniform x in cylinder per source photon
        x=sample_cyl(n)
        vec=x-src; R=np.linalg.norm(vec,axis=1)
        t_in=100.0/vec[:,2]
        Lpath=(1-t_in)*R
        mu=mu_tot(Esrc)
        w=Vc*mu*np.exp(-mu*Lpath)/(Omega*R**2)
        cands.append(np.stack([np.zeros(n), w, Esrc],axis=1))
        pos=entry.copy(); dr=d.copy(); E=Esrc.copy()
        while len(E)>0:
            mu=mu_tot(E)
            s=-np.log(rng.random(len(E)))/mu
            dbox=box_exit(pos,dr)
            inside=s<dbox
            pos=pos[inside]; dr=dr[inside]; E=E[inside]; s=s[inside]; mu=mu[inside]
            if len(E)==0: break
            y=pos+s[:,None]*dr
            xs=sample_cyl(len(E))
            rv=xs-y; r=np.linalg.norm(rv,axis=1); u=rv/r[:,None]
            cth=np.sum(dr*u,axis=1)
            mc=mu_c(E); mR=mu_R(E); mP=mu_pair(E)
            al=E/me
            Ep=E/(1+al*(1-cth))
            muEp=mu_tot(Ep)
            qc=KN_q(E,cth)
            wC=(mc/mu)*Vc*muEp*qc*np.exp(-muEp*r)/r**2
            qR=rayl_q(E,cth)
            wR=(mR/mu)*Vc*mu*qR*np.exp(-mu*r)/r**2
            m511=mu_tot(np.array(0.511))
            wP=(mP/mu)*Vc*2*m511*np.exp(-m511*r)/(4*np.pi*r**2)
            k=len(E)
            cands.append(np.stack([np.ones(k), wC, Ep],axis=1))
            cands.append(np.stack([np.full(k,2.0), wR, E],axis=1))
            cands.append(np.stack([np.full(k,3.0), wP, np.full(k,0.511)],axis=1))
            pc=mc/mu; pR=mR/mu; pP=mP/mu
            uu=rng.random(k)
            isC=uu<pc
            isR=(uu>=pc)&(uu<pc+pR)
            isP=(uu>=pc+pR)&(uu<pc+pR+pP)
            newpos=[];newdir=[];newE=[]
            if isC.any():
                Ei=E[isC]
                epsv=KN_eps(Ei)
                cc=1-(1/epsv-1)/(Ei/me)
                dC=rotate(dr[isC],cc,2*np.pi*rng.random(len(Ei)))
                newpos.append(y[isC]); newdir.append(dC); newE.append(epsv*Ei)
            if isR.any():
                Ei=E[isR]
                cR=rayl_sample(Ei)
                dR=rotate(dr[isR],cR,2*np.pi*rng.random(len(Ei)))
                newpos.append(y[isR]); newdir.append(dR); newE.append(Ei)
            if isP.any():
                kp=isP.sum()
                for _ in range(2):
                    newpos.append(y[isP]); newdir.append(isotropic(kp)); newE.append(np.full(kp,0.511))
            if newE:
                pos=np.concatenate(newpos); dr=np.concatenate(newdir); E=np.concatenate(newE)
                keep=E>Ecut
                pos=pos[keep]; dr=dr[keep]; E=E[keep]
            else:
                pos=np.zeros((0,3)); dr=np.zeros((0,3)); E=np.zeros(0)
    return np.concatenate(cands)

if __name__=="__main__":
    t0=time.time()
    Ns=int(sys.argv[1])
    C=run(Ns)
    np.save('cands.npy',C)
    print('n cands',len(C),'time',time.time()-t0, 'Omega',Omega)
    for ty in [0,1,2,3]:
        m=C[:,0]==ty
        print(ty, m.sum(), C[m,1].sum()/Ns)
'''
open('sim.py','w').write(sim)
import subprocess
r=subprocess.run(['python','sim.py','20000'],capture_output=True,text=True,cwd='.')
print(r.stdout, r.stderr[-3000:])
