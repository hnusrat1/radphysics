
code2 = r'''
import numpy as np
from mc import *
Phi0_hist = 100.0/ (105.0**2)   # per history: 1/(Omega r^2), Omega=0.01 sr (uniform u,v over +-0.05)
Vcyl = np.pi*1.0**2*0.2
def seg_overlap(p0,d,s,zc,R=1.0,h=0.1):
    # overlap length of segment p0 + t d, t in [0,s], with cylinder axis z, radius R, |z-zc|<=h
    dz=d[:,2]; pz=p0[:,2]
    lo=np.zeros_like(s); hi=s.copy()
    small=np.abs(dz)<1e-12
    with np.errstate(divide='ignore',invalid='ignore'):
        ta=(zc-h-pz)/dz; tb=(zc+h-pz)/dz
    t1=np.minimum(ta,tb); t2=np.maximum(ta,tb)
    inside_z_all = (pz>=zc-h)&(pz<=zc+h)
    t1=np.where(small, np.where(inside_z_all,-np.inf,np.inf), t1)
    t2=np.where(small, np.where(inside_z_all,np.inf,-np.inf), t2)
    a=d[:,0]**2+d[:,1]**2
    b=2*(p0[:,0]*d[:,0]+p0[:,1]*d[:,1])
    c=p0[:,0]**2+p0[:,1]**2-R*R
    disc=b*b-4*a*c
    ok=disc>0
    sq=np.sqrt(np.where(ok,disc,0))
    with np.errstate(divide='ignore',invalid='ignore'):
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    aa=a<1e-14
    # radial interval
    r1=np.where(aa, np.where(c<=0,-np.inf,np.inf), np.where(ok,r1,np.inf))
    r2=np.where(aa, np.where(c<=0,np.inf,-np.inf), np.where(ok,r2,-np.inf))
    A=np.maximum(np.maximum(t1,r1),0.0)
    B=np.minimum(np.minimum(t2,r2),s)
    return np.clip(B-A,0,None)

def run(N, depths, seed=1, verbose=True):
    rng=np.random.default_rng(seed)
    zc=100.0+np.array(depths); ncyl=len(zc)
    Eedges=np.arange(0,1.4001,0.05); nEb=len(Eedges)-1
    cedges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
    # tallies per cylinder
    tal={}
    def add(name,shape):
        tal[name]=np.zeros(shape)
    add('Lp_E',(ncyl,nEb)); add('Ls_E',(ncyl,nEb))
    add('Lp_c',(ncyl,10)); add('Ls_c',(ncyl,10))
    add('Lp',(ncyl,)); add('Ls',(ncyl,))
    add('LE_p',(ncyl,)); add('LE_s',(ncyl,))
    for pr in ['p','s']:
        for proc in ['C','PE','PP','R']:
            add(f'I{proc}_{pr}',(ncyl,))   # sum L*mu
            add(f'K{proc}_{pr}',(ncyl,))   # sum L*mu*T (energy to electrons)
    # electron tallies: fine log grid for slowing down and coarse grid
    fine=np.logspace(np.log10(0.001),np.log10(1.4),201)   # 1 keV..1.4 MeV
    eedges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
    ecos=cedges
    Hfine=np.zeros((ncyl,2,len(fine)-1))   # [cyl, parent(0=p,1=s)] histogram of initial electron energies
    Hfine_c=np.zeros((ncyl,len(fine)-1))  # by direction? (not used)
    Hcoarse=np.zeros((ncyl,2,len(eedges)-1))
    Hcos=np.zeros((ncyl,2,10))
    Hcoarse_sum=np.zeros((ncyl,))
    Esum=np.zeros((ncyl,2)); Wsum=np.zeros((ncyl,2))
    # delta-ray / per-electron info
    # initial photon state
    u=rng.uniform(-0.05,0.05,N); v=rng.uniform(-0.05,0.05,N)
    n=np.sqrt(1+u*u+v*v)
    d=np.stack([u/n,v/n,1/n],axis=1)
    pos=np.stack([100*u,100*v,np.full(N,100.0)],axis=1)
    E=np.where(rng.uniform(size=N)<0.5,1.1732,1.3325)
    nint=np.zeros(N,int)
    # Note: photons in pyramid start at z=100 plane (vacuum outside not tracked)
    nsteps=0; totalsc=0
    while pos.shape[0]>0:
        nsteps+=1
        M=pos.shape[0]
        muC=muKN(E); muPE=mu_pe(E); muPP=mu_pair(E); muR=mu_coh(E)
        mut=muC+muPE+muPP+muR
        # exit distance (box x,y in +-15, z<=130)
        with np.errstate(divide='ignore',invalid='ignore'):
            tx=np.where(d[:,0]>0,(15-pos[:,0])/d[:,0],np.where(d[:,0]<0,(-15-pos[:,0])/d[:,0],np.inf))
            ty=np.where(d[:,1]>0,(15-pos[:,1])/d[:,1],np.where(d[:,1]<0,(-15-pos[:,1])/d[:,1],np.inf))
            tz=np.where(d[:,2]>0,(130-pos[:,2])/d[:,2],np.inf)
        sexit=np.minimum(np.minimum(tx,ty),tz)
        sfree=-np.log(rng.uniform(size=M))/mut
        inter=sfree<sexit
        s=np.where(inter,sfree,sexit)
        # scoring
        isp=(nint==0)
        for c in range(ncyl):
            L=seg_overlap(pos,d,s,zc[c])
            m=L>0
            if not m.any(): continue
            idx=np.where(m)[0]
            Lm=L[idx]; Em=E[idx]; dm=d[idx]; pm=isp[idx]
            bE=np.clip(np.digitize(Em,Eedges)-1,0,nEb-1)
            bc=np.clip(np.digitize(dm[:,2],cedges)-1,0,9)
            for flag,name in [(True,'p'),(False,'s')]:
                mm=(pm==flag)
                if not mm.any(): continue
                np.add.at(tal[f'L{name}_E'][c],bE[mm],Lm[mm])
                np.add.at(tal[f'L{name}_c'][c],bc[mm],Lm[mm])
                tal[f'L{name}'][c]+=Lm[mm].sum()
                tal[f'LE_{name}'][c]+=(Lm[mm]*Em[mm]).sum()
            # interactions and electrons
            Em_muC=muC[idx]; Em_PE=muPE[idx]; Em_PP=muPP[idx]; Em_R=muR[idx]
            # Compton virtual sampling
            eps,mus=sample_compton(Em,rng)
            T=Em*(1-eps)
            Ep=Em*eps
            # 3D electron direction
            a=np.where(np.abs(dm[:,0:1])>0.9,np.array([[0,1.0,0]]),np.array([[1.0,0,0]]))
            e1=a-(a*dm).sum(1,keepdims=True)*dm
            e1/=np.linalg.norm(e1,axis=1,keepdims=True)
            e2=np.cross(dm,e1)
            phi=rng.uniform(0,2*np.pi,dm.shape[0])
            kp=mus[:,None]*dm+np.sqrt(np.clip(1-mus**2,0,None))[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
            pe=Em[:,None]*dm-Ep[:,None]*kp
            pen=np.linalg.norm(pe,axis=1)
            cos_e=pe[:,2]/np.maximum(pen,1e-30)
            for flag,name in [(True,'p'),(False,'s')]:
                mm=(pm==flag)
                if not mm.any(): continue
                for cc_ in range(1):
                    pass
                w=Lm[mm]*Em_muC[mm]
                np.add.at(tal[f'IC_{name}'],(c,),w.sum())
                np.add.at(tal[f'KC_{name}'],(c,),(w*T[mm]).sum())
                w2=Lm[mm]*Em_PE[mm]
                tal[f'IPE_{name}'][c]+=w2.sum(); tal[f'KPE_{name}'][c]+=(w2*Em[mm]).sum()
                w3=Lm[mm]*Em_PP[mm]
                tal[f'IPP_{name}'][c]+=w3.sum(); tal[f'KPP_{name}'][c]+=(w3*(Em[mm]-1.022)).sum()
                tal[f'IR_{name}'][c]+=(Lm[mm]*Em_R[mm]).sum()
                # electron spectra (Compton)
                pidx=0 if name=='p' else 1
                Th=T[mm]; wc=w; cosm=cos_e[mm]
                Hfine[c,pidx]+=np.histogram(Th,bins=fine,weights=wc)[0]
                Hcoarse[c,pidx]+=np.histogram(Th,bins=eedges,weights=wc)[0]
                Hcos[c,pidx]+=np.histogram(cosm,bins=cedges,weights=wc)[0]
                Esum[c,pidx]+=(wc*Th).sum(); Wsum[c,pidx]+=wc.sum()
                # photoelectric electrons: T=E, direction = photon direction
                wpe=w2; Tpe=Em[mm]
                Hfine[c,pidx]+=np.histogram(Tpe,bins=fine,weights=wpe)[0]
                Hcoarse[c,pidx]+=np.histogram(Tpe,bins=eedges,weights=wpe)[0]
                Hcos[c,pidx]+=np.histogram(dm[mm,2],bins=cedges,weights=wpe)[0]
                Esum[c,pidx]+=(wpe*Tpe).sum(); Wsum[c,pidx]+=wpe.sum()
                # pair electrons: 2 particles with (E-1.022)/2
                Tp=np.clip((Em[mm]-1.022)/2,0,None)
                wp=2*w3
                Hfine[c,pidx]+=np.histogram(Tp,bins=fine,weights=wp)[0]
                Hcoarse[c,pidx]+=np.histogram(Tp,bins=eedges,weights=wp)[0]
                Hcos[c,pidx]+=np.histogram(dm[mm,2],bins=cedges,weights=wp)[0]
                Esum[c,pidx]+=(wp*Tp).sum(); Wsum[c,pidx]+=wp.sum()
        # --- interactions at end of segment
        if not inter.any() and True:
            pass
        # advance positions
        pos=pos+s[:,None]*d
        # determine process for interacting photons
        ii=np.where(inter)[0]
        if ii.size==0:
            break
        r=rng.uniform(size=ii.size)*mut[ii]
        pC=muC[ii]; pPE=muPE[ii]; pPP=muPP[ii]
        procC=r<pC
        procPE=(~procC)&(r<pC+pPE)
        procPP=(~procC)&(~procPE)&(r<pC+pPE+pPP)
        procR=(~procC)&(~procPE)&(~procPP)
        # Compton
        ic=ii[procC]
        if ic.size>0:
            ee,mm_=sample_compton(E[ic],rng)
            Enew=E[ic]*ee
            # scattered direction
            dd=d[ic]
            a=np.where(np.abs(dd[:,0:1])>0.9,np.array([[0,1.0,0]]),np.array([[1.0,0,0]]))
            e1=a-(a*dd).sum(1,keepdims=True)*dd; e1/=np.linalg.norm(e1,axis=1,keepdims=True)
            e2=np.cross(dd,e1); phi=rng.uniform(0,2*np.pi,ic.size)
            kp=mm_[:,None]*dd+np.sqrt(np.clip(1-mm_**2,0,None))[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
            d[ic]=kp; E[ic]=Enew; nint[ic]+=1
        # Rayleigh
        ir=ii[procR]
        if ir.size>0:
            mu=np.empty(ir.size); todo=np.arange(ir.size)
            while todo.size>0:
                mt=rng.uniform(-1,1,todo.size); acc=rng.uniform(0,1,todo.size)<(1+mt**2)/2
                mu[todo[acc]]=mt[acc]; todo=todo[~acc]
            dd=d[ir]
            a=np.where(np.abs(dd[:,0:1])>0.9,np.array([[0,1.0,0]]),np.array([[1.0,0,0]]))
            e1=a-(a*dd).sum(1,keepdims=True)*dd; e1/=np.linalg.norm(e1,axis=1,keepdims=True)
            e2=np.cross(dd,e1); phi=rng.uniform(0,2*np.pi,ir.size)
            kp=mu[:,None]*dd+np.sqrt(np.clip(1-mu**2,0,None))[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
            d[ir]=kp; nint[ir]+=1
        # pair: kill photon, create two 0.511 photons
        ip=ii[procPP]
        newpos=[];newdir=[];newE=[];newn=[]
        if ip.size>0:
            for _ in range(2):
                z=rng.uniform(-1,1,ip.size); ph=rng.uniform(0,2*np.pi,ip.size)
                sq=np.sqrt(1-z*z)
                newdir.append(np.stack([sq*np.cos(ph),sq*np.sin(ph),z],1))
                newpos.append(pos[ip]); newE.append(np.full(ip.size,0.51099895)); newn.append(np.ones(ip.size,int))
        # kill: PE and PP photons
        alive=np.ones(pos.shape[0],bool)
        alive[ii[procPE|procPP]]=False
        # exits
        alive[np.where(inter==False)[0]]=False
        pos=pos[alive]; d=d[alive]; E=E[alive]; nint=nint[alive]
        # NOTE: newly created photons appended
        if newpos:
            pos=np.concatenate([pos]+newpos); d=np.concatenate([d]+newdir)
            E=np.concatenate([E]+newE); nint=np.concatenate([nint]+newn)
        # position of alive photons: current pos includes advanced position (before alive filter)
        # the filter above was applied after pos advanced; need to ensure alive uses pre-filter indexing
    return dict(tal=tal,Hfine=Hfine,Hcoarse=Hcoarse,Hcos=Hcos,Esum=Esum,Wsum=Wsum,fine=fine,eedges=eedges,zc=zc,nsteps=nsteps)
'''
open('mc2.py','w').write(code2)
print("written")
