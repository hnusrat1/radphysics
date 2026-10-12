
mc2_code = r'''
import numpy as np
from phys import *
from tables import Y_T, R_T
import tables as TB
EDGES_PH=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
EDGES_EL=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
EDGES_CT=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
LINES=np.array([1.1732,1.3325])
DEPTHS=[0.25,0.5,0.75,1.0,5.0]
DET4=4
_tab=np.load('/tmp/work/tables.npz')
_lE=np.log(_tab['E'])
def _interp_tab(name,E):
    return np.interp(np.log(E),_lE,_tab[name])
def perp(u):
    a=np.where(np.abs(u[:,0:1])<0.9, np.array([[1.0,0,0]]), np.array([[0,1.0,0]]))
    e1=np.cross(u,a); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(u,e1)
    return e1,e2
def rotate(u,c,phi):
    e1,e2=perp(u)
    s=np.sqrt(np.maximum(0,1-c*c))
    n=c[:,None]*u+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
    return n/np.linalg.norm(n,axis=1)[:,None]
def sample_KN(E,rng):
    k=E/mc2; L=np.log(1+2*k)
    n=E.size; eps=np.empty(n); cs=np.empty(n)
    todo=np.arange(n)
    while todo.size:
        kk=k[todo]; LL=L[todo]
        e=np.exp(-rng.random(todo.size)*LL)
        c=1-(1/e-1)/kk
        s2=1-c*c
        acc=2*rng.random(todo.size) < (e*e+1-e*s2)
        idx=todo[acc]
        eps[idx]=e[acc]; cs[idx]=c[acc]
        todo=todo[~acc]
    return eps,cs
def kn_dsig_dOmega(E,c):
    k=E/mc2; eps=1/(1+k*(1-c)); s2=1-c*c
    return 0.5*re**2*eps**2*(eps+1/eps-s2)
def isotropic(n,rng):
    v=rng.normal(size=(n,3)); return v/np.linalg.norm(v,axis=1)[:,None]
DELTA=0.01
def run(N,seed,depths=DEPTHS):
    rng=np.random.default_rng(seed)
    tx=rng.uniform(-0.05,0.05,N); ty=rng.uniform(-0.05,0.05,N)
    acc=rng.random(N) < (1+tx*tx+ty*ty)**-1.5
    tx,ty=tx[acc],ty[acc]; N=tx.size
    E0=LINES[rng.integers(0,2,N)]
    d=np.stack([tx,ty,np.ones(N)],1); d/=np.linalg.norm(d,axis=1)[:,None]
    pos=np.stack([100*tx,100*ty,np.zeros(N)],1)+1e-9*d
    act_pos=pos; act_dir=d; act_E=E0.copy()
    nd=len(depths)
    S={}
    S['N']=N
    S['hE']=np.zeros((nd,28)); S['hC']=np.zeros((nd,10))
    S['g']=np.zeros(nd); S['gE']=np.zeros(nd)
    S['int']=np.zeros((nd,4))   # g*mu_i(Ea): C,PE,pair,R
    S['trE']=np.zeros((nd,4))   # g*mu_i*energy transferred (C,PE,pair,R)
    S['kcol']=np.zeros(nd); S['kgross']=np.zeros(nd)
    S['kcol_scat_C']=np.zeros(nd)
    # electron tallies at DET4
    S['L3_hT']=np.zeros(len(EDGES_EL)-1); S['L3_w']=0.0; S['L3_wT']=0.0
    S['L3_hC']=np.zeros(10)
    S['L3_hC_T']=np.zeros(10)   # not used
    S['Q0_hEa']=np.zeros(0)
    S['Q0_edges']=np.linspace(0,1.4,701)
    S['Q0_w']=np.zeros(700)     # weights of photon electrons vs parent energy (Compton)
    S['Q0_wpe']=np.zeros(700)   # photo electrons vs energy
    S['L4_hC']=np.zeros(10)     # track-weighted photon-electron angle
    S['D4_hC']=np.zeros(10)     # delta first gen angle
    S['Lpk']=0.0
    S['Wpe_T']=[]
    ncoll=0; it=0
    while act_E.size>0 and it<80:
        it+=1
        n=act_E.size
        with np.errstate(divide='ignore',invalid='ignore'):
            tx_=np.where(act_dir[:,0]>0,(15-act_pos[:,0])/act_dir[:,0],np.where(act_dir[:,0]<0,(-15-act_pos[:,0])/act_dir[:,0],np.inf))
            ty_=np.where(act_dir[:,1]>0,(15-act_pos[:,1])/act_dir[:,1],np.where(act_dir[:,1]<0,(-15-act_pos[:,1])/act_dir[:,1],np.inf))
            tz_=np.where(act_dir[:,2]>0,(30-act_pos[:,2])/act_dir[:,2],np.where(act_dir[:,2]<0,(0-act_pos[:,2])/act_dir[:,2],np.inf))
        texit=np.maximum(np.minimum(np.minimum(tx_,ty_),tz_),0)
        mu=mu_tot(act_E)
        s=-np.log(rng.random(n))/mu
        inside=s<texit
        idx=np.nonzero(inside)[0]
        if idx.size==0: break
        E_=act_E[idx]; u=act_dir[idx]; x=act_pos[idx]+s[idx,None]*u; mu_=mu[idx]
        ncoll+=idx.size
        pc=mu_inc(E_)/mu_; pr=mu_coh(E_)/mu_; pe=mu_pe(E_)/mu_; pp=mu_pair(E_)/mu_
        mC=mu_inc(E_); mPE=mu_pe(E_); mPP=mu_pair(E_); mR=mu_coh(E_)
        for j,dz in enumerate(depths):
            dvec=np.array([0,0,dz])[None,:]-x
            R=np.linalg.norm(dvec,axis=1); nh=dvec/R[:,None]
            c=np.sum(u*nh,axis=1)
            k=E_/mc2; ep=E_/(1+k*(1-c))
            p=kn_dsig_dOmega(E_,c)/kn_total(E_)
            gC=pc*p*np.exp(-mu_tot(ep)*R)/R**2
            pR=3*(1+c*c)/(16*np.pi)
            gR=pr*pR*np.exp(-mu_*R)/R**2
            gA=pp*2/(4*np.pi)*np.exp(-mu_tot(0.511)*R)/R**2
            cb=np.clip(np.searchsorted(EDGES_CT,nh[:,2],side='right')-1,0,9)
            for typ,g,Ea in [(0,gC,ep),(1,gR,E_),(2,gA,np.full(E_.size,0.511))]:
                sel=g>0
                if not sel.any(): continue
                gg=g[sel]; ee=Ea[sel]
                pb=np.clip(np.searchsorted(EDGES_PH,ee,side='right')-1,0,27)
                S['hE'][j]+=np.bincount(pb,weights=gg,minlength=28)
                S['hC'][j]+=np.bincount(cb[sel],weights=gg,minlength=10)
                S['g'][j]+=gg.sum(); S['gE'][j]+=(gg*ee).sum()
                mCa=mu_inc(ee); mPEa=mu_pe(ee); mPPa=mu_pair(ee); mRa=mu_coh(ee)
                S['int'][j]+=np.array([(gg*mCa).sum(),(gg*mPEa).sum(),(gg*mPPa).sum(),(gg*mRa).sum()])
                ftr=_interp_tab('ftr',ee); kcf=_interp_tab('kcf',ee)
                trC=(gg*mCa*ee*ftr).sum(); trPE=(gg*mPEa*ee).sum(); trPP=(gg*mPPa*np.maximum(ee-2*0.511,0)).sum()
                S['trE'][j]+=np.array([trC,trPE,trPP,0.0])
                kC=(gg*mCa*ee*kcf).sum()
                kPE=(gg*mPEa*ee*(1-Y_T(ee))).sum()
                kPP=(gg*mPPa*np.maximum(ee-1.022,0)*(1-Y_T(np.maximum(ee-1.022,0.01)/2))).sum()
                S['kcol'][j]+=kC+kPE+kPP
                S['kgross'][j]+=trC+trPE+trPP
                if typ==0 and False: pass
            if j==DET4:
                # arrivals: all types; electrons from Compton / PE / pair
                uA=nh
                Ea_all=[];gw=[]
                # collect arrivals (g>0) with their energies: recompute for all three types
                for typ,g,Ea in [(0,gC,ep),(1,gR,E_),(2,gA,np.full(E_.size,0.511))]:
                    sel=np.nonzero(g>0)[0]
                    if sel.size==0: continue
                    gg=g[sel]; ee=Ea[sel]; uu=uA[sel]
                    wC=gg*mu_inc(ee); wPE=gg*mu_pe(ee); wPP=gg*mu_pair(ee)
                    # Compton electrons
                    eps,cs=sample_KN(ee,rng)
                    phi=rng.uniform(0,2*np.pi,sel.size)
                    nout=rotate(uu,cs,phi)
                    T=ee*(1-eps)
                    pe_=ee[:,None]*uu-(ee*eps)[:,None]*nout
                    cos_e=pe_[:,2]/np.linalg.norm(pe_,axis=1)
                    hT=np.histogram(T,bins=EDGES_EL,weights=wC)[0]
                    S['L3_hT']+=hT
                    S['L3_w']+=wC.sum(); S['L3_wT']+=(wC*T).sum()
                    cbe=np.clip(np.searchsorted(EDGES_CT,cos_e,side='right')-1,0,9)
                    S['L3_hC']+=np.bincount(cbe,weights=wC,minlength=10)
                    # L4 track-weighted photon electron angles
                    Lp=np.maximum(R_T(np.maximum(T,1e-4))-R_T(DELTA),0)*(T>=DELTA)
                    S['L4_hC']+=np.bincount(cbe,weights=wC*Lp,minlength=10)
                    # delta first generation (1/T^2 law), parent T
                    ok=(T>=2*DELTA)
                    if ok.any():
                        T0=T[ok]; wL=wC[ok]*Lp[ok]; cp=cos_e[ok]
                        beta2=1-1/(1+T0/mc2)**2
                        Nrate=2*np.pi*re**2*mc2*ne/beta2*(1/DELTA-2/T0)
                        u_=rng.random(T0.size)
                        Td=1/(1/DELTA-u_*(1/DELTA-2/T0))
                        Ld=np.maximum(R_T(Td)-R_T(DELTA),0)
                        wd=wL*Nrate*Ld
                        cr=np.sqrt(np.clip(Td*(T0+2*mc2)/(T0*(Td+2*mc2)),0,1))
                        phi2=rng.uniform(0,2*np.pi,T0.size)
                        cbeam=cp*cr+np.sqrt(np.maximum(0,1-cp*cp))*np.sqrt(np.maximum(0,1-cr*cr))*np.cos(phi2)
                        cbd=np.clip(np.searchsorted(EDGES_CT,cbeam,side='right')-1,0,9)
                        S['D4_hC']+=np.bincount(cbd,weights=wd,minlength=10)
                    # photo/pair electrons
                    if typ in (0,1,2):
                        if (wPE>0).any():
                            hh=np.histogram(ee,bins=EDGES_EL,weights=wPE)[0]
                            S['L3_hT']+=hh; S['L3_w']+=wPE.sum(); S['L3_wT']+=(wPE*ee).sum()
                            cbp=np.clip(np.searchsorted(EDGES_CT,uu[:,2],side='right')-1,0,9)
                            S['L3_hC']+=np.bincount(cbp,weights=wPE,minlength=10)
                        Tp=(ee-1.022)/2
                        okp=Tp>0
                        if okp.any():
                            hp=np.histogram(Tp[okp],bins=EDGES_EL,weights=2*wPP[okp])[0]
                            S['L3_hT']+=hp; S['L3_w']+=2*wPP[okp].sum(); S['L3_wT']+=(2*wPP[okp]*Tp[okp]).sum()
                    # source for Q0 (Compton electrons vs parent energy), E-histogram
                    S['Q0_w']+=np.histogram(ee,bins=S['Q0_edges'],weights=wC)[0]
                    S['Q0_wpe']+=np.histogram(ee,bins=S['Q0_edges'],weights=wPE)[0]
        # interaction sampling
        u2=rng.random(idx.size)
        isC=u2<pc
        isR=(~isC)&(u2<pc+pr)
        isPE=(~isC)&(~isR)&(u2<pc+pr+pe)
        isPP=(~isC)&(~isR)&(~isPE)
        newpos=[];newdir=[];newE=[]
        ci=np.nonzero(isC)[0]
        if ci.size:
            eps,cs=sample_KN(E_[ci],rng)
            phi=rng.uniform(0,2*np.pi,ci.size)
            newpos.append(x[ci]); newdir.append(rotate(u[ci],cs,phi)); newE.append(E_[ci]*eps)
        ri=np.nonzero(isR)[0]
        if ri.size:
            cc=np.empty(ri.size); todo=np.arange(ri.size)
            while todo.size:
                cr=rng.uniform(-1,1,todo.size); a=rng.random(todo.size)<(1+cr*cr)/2
                cc[todo[a]]=cr[a]; todo=todo[~a]
            phi=rng.uniform(0,2*np.pi,ri.size)
            newpos.append(x[ri]); newdir.append(rotate(u[ri],cc,phi)); newE.append(E_[ri])
        pi_=np.nonzero(isPP)[0]
        if pi_.size:
            for _ in range(2):
                newpos.append(x[pi_]); newdir.append(isotropic(pi_.size,rng)); newE.append(np.full(pi_.size,0.511))
        if newE:
            ne_=np.concatenate(newE); nd_=np.concatenate(newdir); npos=np.concatenate(newpos)
            keep=ne_>0.005
            act_E=ne_[keep]; act_dir=nd_[keep]; act_pos=npos[keep]
        else:
            act_E=np.zeros(0); act_dir=np.zeros((0,3)); act_pos=np.zeros((0,3))
    S['ncoll']=ncoll
    return S
'''
open('/tmp/work/mc2.py','w').write(mc2_code)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, mc2, time
t=time.time(); S=mc2.run(100000,5); print(time.time()-t)
print(S['g'], S['N'], S['L3_w'], S['L3_wT']/S['L3_w'])
