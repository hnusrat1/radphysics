
code = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import xs, mell
from mell import ME, DCUT, sig_delta, sample_delta_eps
X0=36.08
EB25=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
ZB=np.arange(-4,4.0001,0.02)
def run_e(births, Btot, n, seed):
    rng=np.random.default_rng(seed)
    w=births[:,3]; cw=np.cumsum(w); cw/=cw[-1]
    ii=np.searchsorted(cw, rng.uniform(0,1,n)); ii=np.clip(ii,0,len(w)-1)
    T=births[ii,0].copy(); c=births[ii,1].copy(); F=births[ii,2].copy(); kind=births[ii,4].copy()
    T0s=T.sum()
    wgt=Btot/n
    zrel=np.zeros(n); pos=(kind==3); dl=np.zeros(n,bool); Fp=F.copy(); wg=np.full(n,wgt)
    kern=np.zeros(len(ZB)-1)
    # sub-cut births: deposit at 0
    low=T<DCUT
    kern+=np.histogram(np.zeros(low.sum()),bins=ZB,weights=T[low])[0]
    keep=~low
    T=T[keep]; c=c[keep]; zrel=zrel[keep]; pos=pos[keep]; dl=dl[keep]; Fp=Fp[keep]; wg=wg[keep]
    # tallies
    fl_bin=np.zeros(len(EB25)-1); fl_tot=0.0; fl_T=0.0; fl_del=0.0; ang=np.zeros(10)
    rad=0.0; col=0.0; dep_tot=0.0; 
    cedges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0001])
    fl_bin_sc=np.zeros(len(EB25)-1)
    fl_tot_sc=0.0; ntr=0
    steps=0
    while len(T)>0:
        steps+=1
        dT=np.minimum(np.maximum(0.02*T,2e-5),T)
        St=xs.Stot(T); Sc=xs.Scol(T); Sr=xs.Srad(T)
        ds=dT/St
        Tmid=T-dT/2
        track=~pos
        # tallies (electrons only, not positrons)
        wd=wg*ds
        fl_tot+=wd[track].sum(); fl_T+=(wd*Tmid)[track].sum(); fl_del+=wd[track&dl].sum()
        b=np.clip(np.searchsorted(EB25,Tmid,side='right')-1,0,len(EB25)-2)
        inb=(Tmid>=EB25[0])&(Tmid<EB25[-1])&track
        np.add.at(fl_bin,b[inb],wd[inb])
        cb=np.clip(np.searchsorted(cedges,c,side='right')-1,0,9)
        np.add.at(ang,cb[track],wd[track])
        rad+=(wd*Sr)[track].sum(); col+=(wd*Sc)[track].sum()
        # deltas
        lam=sig_delta(T)*ds
        nd=rng.poisson(lam)
        parents=np.repeat(np.arange(len(T)),nd)
        epsum=np.zeros(len(T))
        if len(parents):
            Tp=T[parents]
            eps=sample_delta_eps(Tp,rng.uniform(0,1,len(parents)))
            epsum=np.bincount(parents,weights=eps,minlength=len(T))
            cth=np.sqrt(np.clip(eps*(Tp+2*ME)/(Tp*(eps+2*ME)),0,1))
            cp=c[parents]
            phi=rng.uniform(0,2*np.pi,len(parents))
            cd=cp*cth+np.sqrt(np.clip(1-cp**2,0,1))*np.sqrt(np.clip(1-cth**2,0,1))*np.cos(phi)
        # parent deposit
        zmid=zrel+0.5*c*ds
        depp=np.clip(Sc/St*dT-epsum,0,None)
        dep_tot+=depp.sum()
        kern+=np.histogram(zmid,bins=ZB,weights=depp)[0]
        # angular Highland
        p=np.sqrt(T*(T+2*ME)); beta=p/(T+ME)
        xr=np.maximum(ds/X0,1e-9)
        th0=13.6/(beta*p)*np.sqrt(xr)*(1+0.038*np.log(xr))
        th=th0*np.sqrt(-2*np.log(np.maximum(rng.uniform(0,1,len(T)),1e-300)))
        cth_s=np.cos(th); sth=np.sin(th)
        iso=th0>1.2
        phi2=rng.uniform(0,2*np.pi,len(T))
        cn=c*cth_s+np.sqrt(np.clip(1-c**2,0,1))*sth*np.cos(phi2)
        cn=np.where(iso,rng.uniform(-1,1,len(T)),cn)
        cn=np.clip(cn,-1,1)
        cmid=0.5*(c+cn)
        zrel=zrel+cmid*ds
        Tn=T-dT
        c=cn
        # below cutoff: deposit remaining energy at current pos, deactivate
        done=Tn<DCUT
        if done.any():
            kern+=np.histogram(zrel[done],bins=ZB,weights=np.clip(Tn[done],0,None))[0]
            dep_tot+=np.clip(Tn[done],0,None).sum()
        # build next population
        alive=~done
        T_=Tn[alive]; c_=c[alive]; z_=zrel[alive]; pos_=pos[alive]; dl_=dl[alive]; Fp_=Fp[alive]; wg_=wg[alive]
        if len(parents):
            # new deltas
            pd=parents
            okd=(pd>=0)
            Td=eps; cdn=cd; zd=zmid[pd]; wgd=wg[pd]; Fd=Fp[pd]
            T_=np.concatenate([T_,Td[alive[pd]] if False else Td]); 
            c_=np.concatenate([c_,cdn]); z_=np.concatenate([z_,zd]); pos_=np.concatenate([pos_,np.zeros(len(pd),bool)])
            dl_=np.concatenate([dl_,np.ones(len(pd),bool)]); Fp_=np.concatenate([Fp_,Fd]); wg_=np.concatenate([wg_,wgd])
            # deltas spawned from dead parents are still valid (they were spawned before deactivation)
        T,c,zrel,pos,dl,Fp,wg=T_,c_,z_,pos_,dl_,Fp_,wg_
    return dict(fl_bin=fl_bin,fl_tot=fl_tot,fl_T=fl_T,fl_del=fl_del,ang=ang,rad=rad,col=col,kern=kern,
                T0s=T0s,n=n,steps=steps,dep_tot=dep_tot)
'''
open('/tmp/work/elec.py','w').write(code)
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, pickle, time, elec
tot=pickle.load(open('/tmp/work/tot.pkl','rb'))
B=tot['births']; N=tot['N']
VCYL=np.pi*0.2; PHI=1/(0.0099751*110**2)
Ball=B[:,3].sum()/(N*VCYL*PHI)
print('birth density per Phi0 (all kinds):',Ball)
t=time.time(); res=elec.run_e(B,Ball,20000,1); print(time.time()-t, res['steps'])
print(res['fl_tot'], res['fl_del']/res['fl_tot'], res['rad']/res['col'], res['dep_tot']/ (res['T0s']))
