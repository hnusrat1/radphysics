
mc2 = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import xs, mcph
from mcph import ME, rotate, rayleigh_mu, ortho
PHI0_PER_SRC=mcph.PHI0_PER_SRC; VCYL=mcph.VCYL
# InvCDF table for KN in mu
NQ=2000
QG=(np.arange(NQ)+0.5)/NQ
INVC=np.array([np.interp(QG,mcph.CDFG[i],mcph.MUG) for i in range(len(mcph.EG))])
LEG=np.log(mcph.EG)
def kn_mu(E,u):
    lE=np.log(np.clip(E,mcph.EG[0],mcph.EG[-1]))
    x=(lE-LEG[0])/(LEG[-1]-LEG[0])*(len(mcph.EG)-1)
    i0=np.clip(np.floor(x).astype(int),0,len(mcph.EG)-2); f=x-i0
    x2=u*NQ-0.5; q0=np.clip(np.floor(x2).astype(int),0,NQ-2); g=np.clip(x2-q0,0,1)
    a=INVC[i0,q0]*(1-g)+INVC[i0,q0+1]*g
    b=INVC[i0+1,q0]*(1-g)+INVC[i0+1,q0+1]*g
    return a*(1-f)+b*f
# f_KN mean energy-transfer fraction table
FKN=np.array([np.trapz(
    (lambda e: (1/(1+e*(1-mcph.MUG)))**2*((1/(1+e*(1-mcph.MUG)))+(1+e*(1-mcph.MUG)) -(1-mcph.MUG**2))*(1-1/(1+e*(1-mcph.MUG))))(E/ME),mcph.MUG)
    /np.trapz((lambda e: (1/(1+e*(1-mcph.MUG)))**2*((1/(1+e*(1-mcph.MUG)))+(1+e*(1-mcph.MUG)) -(1-mcph.MUG**2)))(E/ME),mcph.MUG) for E in mcph.EG])
def fkn(E): return np.interp(np.log(np.clip(E,mcph.EG[0],mcph.EG[-1])),LEG,FKN)
def chord(P,D,tmax,R,za,zb):
    # overlap length in t in [0,tmax] of radial cylinder R (axis z) and slab za<=z<=zb ; za,zb arrays or scalars
    n=len(P)
    dz=D[:,2]; z0=P[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(za-z0)/dz; t2=(zb-z0)/dz
    lo=np.minimum(t1,t2); hi=np.maximum(t1,t2)
    zpar=np.abs(dz)<1e-12
    inside=(z0>=za)&(z0<=zb)
    lo=np.where(zpar,np.where(inside,-np.inf,np.inf),lo)
    hi=np.where(zpar,np.where(inside,np.inf,-np.inf),hi)
    a=D[:,0]**2+D[:,1]**2; b=2*(P[:,0]*D[:,0]+P[:,1]*D[:,1]); c=P[:,0]**2+P[:,1]**2-R**2
    disc=b*b-4*a*c
    with np.errstate(divide='ignore',invalid='ignore'):
        sq=np.sqrt(np.maximum(disc,0))
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    ok=(a>1e-14)&(disc>=0)
    rlo=np.where(ok,np.minimum(r1,r2),np.where(a>1e-14,np.inf,np.where(c<=0,-np.inf,np.inf)))
    rhi=np.where(ok,np.maximum(r1,r2),np.where(a>1e-14,-np.inf,np.where(c<=0,np.inf,-np.inf)))
    Llo=np.maximum.reduce([np.zeros(n),lo,rlo]); Lhi=np.minimum.reduce([np.full(n,0.0)+tmax,hi,rhi])
    return np.maximum(Lhi-Llo,0.0), Llo, Lhi
def run(N, seed, nbins_q=48):
    rng=np.random.default_rng(seed)
    # source directions uniform in solid angle within 10x10 pyramid
    acc=[]; n=0
    while n<N:
        m=int(N*1.1)+10
        a=rng.uniform(-0.05,0.05,m); b=rng.uniform(-0.05,0.05,m)
        ok=rng.uniform(0,1,m)<(1+a*a+b*b)**(-1.5)
        acc.append(np.stack([a[ok],b[ok]],1)); n+=ok.sum()
    ab=np.concatenate(acc)[:N]
    D=np.column_stack([ab[:,0],ab[:,1],np.ones(N)]); D/=np.linalg.norm(D,axis=1)[:,None]
    t0=100.0/D[:,2]
    P=np.column_stack([D[:,0]*t0,D[:,1]*t0,np.zeros(N)-0.0])
    P[:,2]=0.0
    E=np.full(N,2.0); F=np.zeros(N,dtype=np.int8)
    # tallies
    NE=len(mcph.EB)-1
    flu_E=np.zeros((2,NE)); flu_c=np.zeros((2,10)); en_fl=np.zeros(2)
    intr={k:np.zeros(2) for k in ['coh','inc','pe','pr']}
    etr={k:np.zeros(2) for k in ['coh','inc','pe','pr']}
    Qslab=np.zeros(nbins_q)
    births=[]  # arrays T0,c,F,w,kind
    slab_edges=np.arange(nbins_q+1)*0.25
    cedges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0001])
    # annihilation etc appended later
    Nphot=0
    it=0
    while len(E)>0:
        it+=1
        # exit distance
        with np.errstate(divide='ignore',invalid='ignore'):
            tx=np.where(D[:,0]>0,(15-P[:,0])/D[:,0],np.where(D[:,0]<0,(-15-P[:,0])/D[:,0],np.inf))
            ty=np.where(D[:,1]>0,(15-P[:,1])/D[:,1],np.where(D[:,1]<0,(-15-P[:,1])/D[:,1],np.inf))
            tz=np.where(D[:,2]>0,(30-P[:,2])/D[:,2],np.where(D[:,2]<0,(0-P[:,2])/D[:,2],np.inf))
        texit=np.minimum(np.minimum(tx,ty),tz)
        texit=np.maximum(texit,0)
        mc_={k:xs.mu(E,k) for k in ['coh','inc','pe','pr']}
        mt=sum(mc_.values())
        s=-np.log(rng.uniform(0,1,len(E)))/mt
        inter=s<texit
        move=np.minimum(s,texit)
        # cylinder tally
        Lc,_,_=chord(P,D,move,1.0,9.9,10.1)
        sel=Lc>0
        if sel.any():
            Es=E[sel]; Fs=F[sel]; Ls=Lc[sel]; cs=D[sel,2]
            idx=np.clip(np.floor(Es/0.01).astype(int),0,NE-1)
            for f in (0,1):
                m=Fs==f
                if m.any():
                    np.add.at(flu_E[f],idx[m],Ls[m])
                    flu_c[f]+=np.histogram(cs[m],bins=cedges,weights=Ls[m])[0]
                    en_fl[f]+=np.sum(Ls[m]*Es[m])
                    for k in mc_:
                        mk=mc_[k][sel][m]
                        intr[k][f]+=np.sum(Ls[m]*mk)
                    ftrc=fkn(Es[m])
                    etr['inc'][f]+=np.sum(Ls[m]*mc_['inc'][sel][m]*Es[m]*ftrc)
                    etr['pe'][f]+=np.sum(Ls[m]*mc_['pe'][sel][m]*Es[m])
                    etr['pr'][f]+=np.sum(Ls[m]*mc_['pr'][sel][m]*np.maximum(Es[m]-1.022,0))
            # births from segments
            Pf=E[sel]; cs_=cs; Fsel=Fs
            u=rng.uniform(0,1,len(Es)); phi=rng.uniform(0,2*np.pi,len(Es))
            # Compton
            wC=Ls*mc_['inc'][sel]
            mu_=kn_mu(Es,rng.uniform(0,1,len(Es)))
            Ep=Es/(1+Es/ME*(1-mu_))
            dpz=mu_*cs_+np.sqrt(np.maximum(0,1-mu_**2))*np.sqrt(np.maximum(0,1-cs_**2))*np.cos(phi)
            pez=Es*cs_-Ep*dpz
            pe_mag=np.sqrt(np.maximum(Es**2+Ep**2-2*Es*Ep*mu_,1e-30))
            cE=pez/pe_mag
            births.append(np.column_stack([Es-Ep,cE,Fsel,wC,np.zeros(len(Es))]))
            # photo
            wP=Ls*mc_['pe'][sel]
            births.append(np.column_stack([Es,cs_,Fsel,wP,np.ones(len(Es))]))
            # pair: e- share uniform, e+ rest
            wR=Ls*mc_['pr'][sel]
            ok=Es>1.022
            Tk=(Es-1.022)
            sh=rng.uniform(0,1,len(Es))
            births.append(np.column_stack([np.where(ok,Tk*sh,0),cs_,Fsel,wR,2*np.ones(len(Es))]))
            births.append(np.column_stack([np.where(ok,Tk*(1-sh),0),cs_,Fsel,wR,3*np.ones(len(Es))]))
            # Q slabs (radius 3 cm): radial chord then z-slabs
            Lr,Llo,Lhi=chord(P[sel],D[sel],move[sel],3.0,-1e9,1e9)
            zs=P[sel,2]+Llo*D[sel,2]; ze=P[sel,2]+Lhi*D[sel,2]
            zlo=np.minimum(zs,ze); zhi=np.maximum(zs,ze)
            ok2=(Lr>0)&(np.abs(D[sel,2])>1e-9)
            wq=Ls*0+ (Es*mc_['inc'][sel]*fkn(Es)*Es/Es + mc_['pe'][sel]*Es+mc_['pr'][sel]*np.maximum(Es-1.022,0))
            # for each slab j, overlap in z -> t-length = zlen/|cos|
            jlo=np.clip(np.floor(zlo/0.25).astype(int),0,nbins_q)
            jhi=np.clip(np.floor(zhi/0.25).astype(int),-1,nbins_q-1)
            for j in range(max(0,int(jlo[ok2].min()) if ok2.any() else 0), min(nbins_q-1,int(jhi[ok2].max()) if ok2.any() else 0)+1):
                za=j*0.25; zb=za+0.25
                ov=np.minimum(zhi,zb)-np.maximum(zlo,za)
                ov=np.where(ok2&(ov>0),ov,0)
                # track length = z-overlap/|cos|; only radial-part fraction handled by z-span already (radial clipped)
                tl=ov/np.maximum(np.abs(D[sel,2]),1e-9)
                # weight by energy transfer rate
                Qslab[j]+=np.sum(tl*wq)
        # interactions: process
        Pn=P+move[:,None]*D
        # move photons
        if inter.any():
            ii=np.where(inter)[0]
            Ei=E[ii]; Di=D[ii]; Fi=F[ii]
            r=rng.uniform(0,1,len(ii))*mt[ii]
            pc=mc_['coh'][ii]; pi=pc+mc_['inc'][ii]; pp=pi+mc_['pe'][ii]
            isc=r<pc; isi=(r>=pc)&(r<pi); ipe=(r>=pi)&(r<pp); ipr=r>=pp
            # coherent
            if isc.any():
                k=ii[isc]
                mu_r=rayleigh_mu(len(k),rng)
                D[k]=rotate(D[k],mu_r,rng.uniform(0,2*np.pi,len(k)))
                F[k]=1
            if isi.any():
                k=ii[isi]
                mu_=kn_mu(E[k],rng.uniform(0,1,len(k)))
                Ep=E[k]/(1+E[k]/ME*(1-mu_))
                D[k]=rotate(D[k],mu_,rng.uniform(0,2*np.pi,len(k)))
                E[k]=Ep; F[k]=1
            # photo and pair kill
            kill=np.zeros(len(E),bool)
            kill[ii[ipe]]=True
            kp=ii[ipr]
            if len(kp):
                kill[kp]=True
                # annihilation photons (two, back to back, isotropic)
                nd=rng.normal(size=(len(kp),3)); nd/=np.linalg.norm(nd,axis=1)[:,None]
                newP=np.concatenate([Pn[kp],Pn[kp]]); newD=np.concatenate([nd,-nd])
                newE=np.full(2*len(kp),0.511); newF=np.ones(2*len(kp),dtype=np.int8)
            else:
                newP=np.zeros((0,3)); newD=np.zeros((0,3)); newE=np.zeros(0); newF=np.zeros(0,dtype=np.int8)
        else:
            kill=np.zeros(len(E),bool)
            newP=np.zeros((0,3)); newD=np.zeros((0,3)); newE=np.zeros(0); newF=np.zeros(0,dtype=np.int8)
        # kill exits
        kill|=(~inter)
        # tally interaction-type counts for process (expected) done by segments
        # update positions of survivors
        keep=~kill
        P=Pn[keep]; D=D[keep]; E=E[keep]; F=F[keep]
        # ensure exits removed: those with ~inter killed above
        # include annihilation photons
        if len(newE):
            P=np.concatenate([P,newP]); D=np.concatenate([D,newD]); E=np.concatenate([E,newE]); F=np.concatenate([F,newF])
        # kill photons below 1 keV (absorbed effectively)
        alive=E>0.001
        P=P[alive]; D=D[alive]; E=E[alive]; F=F[alive]
    Bt=np.vstack(births)
    return dict(flu_E=flu_E,flu_c=flu_c,en_fl=en_fl,intr=intr,etr=etr,Qslab=Qslab,births=Bt)
'''
open('/tmp/work/mcph2.py','w').write(mc2)
import importlib, time
import mcph2; importlib.reload(mcph2)
t=time.time(); r=mcph2.run(20000,1); print(time.time()-t)
print(r['flu_E'].sum(axis=1), r['intr'], r['births'].shape)
