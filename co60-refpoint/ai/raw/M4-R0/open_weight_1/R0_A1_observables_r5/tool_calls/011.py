import textwrap, os
code = r'''
import numpy as np, time, math

mec2 = 0.51099895
r_e = 2.8179403262e-13
Na  = 6.02214076e23
molW = 18.01528
e_per_g_water = 10.0/molW*Na
mol_per_g_water = Na/molW
I_water = 78e-6
I_air   = 85.7e-6
ZA_water = 10.0/18.01528
ZA_air = 0.4992

def KN_barn(E):
    g = E/mec2
    a = 1+2*g
    return (2*np.pi*r_e**2*((1+g)/g**2*(2*(1+g)/a-np.log(a)/g)+np.log(a)/(2*g)-(1+3*g)/a**2))/1e-24

def mu_inc_w(E): return KN_barn(E)*e_per_g_water/1e24
def mu_coh_w(E): return 0.00085*(1.25/np.asarray(E,float))**0.65
def mu_pho_w(E): return 1.2e-5*np.asarray(E,float)**-3.0
def mu_pai_w(E):
    E=np.asarray(E,float); out=np.zeros_like(E); m=E>1.022
    x=(E[m]-1.022)/9.0
    out[m]=(0.114+2*0.0018)*x*x*1e-24*mol_per_g_water
    return out
def mu_tot_w(E): return mu_inc_w(E)+mu_coh_w(E)+mu_pho_w(E)+mu_pai_w(E)
def mu_process(E):
    inc=mu_inc_w(E); coh=mu_coh_w(E); pho=mu_pho_w(E); pai=mu_pai_w(E); tot=inc+coh+pho+pai
    return inc/tot,coh/tot,pho/tot,pai/tot

Kc=0.153536
def s_col(E,ZA,I):
    E=np.asarray(E,float); tau=E/mec2; gam=tau+1.0; beta2=1.0-1.0/gam**2
    B=np.log(2*mec2*beta2*gam**2*(tau/2.0*mec2)/I**2)-beta2-(2*gam-1)/gam**2*math.log(2)+(gam-1)**2/(8*gam**2)
    return Kc*ZA/beta2*B

def l_delta(E,ZA,I,delta=0.01):
    E=np.asarray(E,float); tau=E/mec2; gam=tau+1.0; beta2=1.0-1.0/gam**2
    B=np.log(2*mec2*beta2*gam**2*delta/I**2)-beta2-(2*gam-1)/gam**2*math.log(2)+(gam-1)**2/(8*gam**2)
    return Kc*ZA/beta2*B

def s_rad_w(E):
    E=np.asarray(E,float)
    A=0.007/(3.667*(1+mec2))
    return A*3.667*(E+mec2)

phE_bins=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
elE_bins=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
ct_bins=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

def basis(u):
    zhat=np.array([0.,0.,1.])
    c=np.cross(u,zhat)
    n=np.linalg.norm(c,axis=1)
    e1=c.copy()
    small=n<1e-4
    if small.any():
        c2=np.cross(u[small],np.array([1.,0.,0.]))
        n2=np.linalg.norm(c2,axis=1)
        c2=c2/n2[:,None]
        e1[small]=c2
        n[small]=1.0
    e1=e1/n[:,None]
    e2=np.cross(u,e1)
    return e1,e2

def track_in_cylinder(r0,r1,R=1.0,zmin=4.9,zmax=5.1):
    d=r1-r0
    dz=d[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        tz1=(zmin-r0[:,2])/dz; tz2=(zmax-r0[:,2])/dz
    tza=np.minimum(tz1,tz2); tzb=np.maximum(tz1,tz2)
    a=d[:,0]**2+d[:,1]**2
    b=2*(r0[:,0]*d[:,0]+r0[:,1]*d[:,1])
    cc=r0[:,0]**2+r0[:,1]**2-R**2
    disc=b*b-4*a*cc
    tr1=np.full_like(a,-1e9); tr2=np.full_like(a,1e9)
    ok=a>1e-12
    sq=np.sqrt(np.maximum(disc,0))
    tr1[ok]=(-b[ok]-sq[ok])/(2*a[ok])
    tr2[ok]=(-b[ok]+sq[ok])/(2*a[ok])
    inside1=(a<=1e-12)&(cc<=0)
    tra=np.minimum(tr1,tr2); trb=np.maximum(tr1,tr2)
    lo=np.maximum(tza,tra); hi=np.minimum(tzb,trb)
    lo=np.where(inside1,tza,lo); hi=np.where(inside1,tzb,hi)
    track=np.maximum(0.,hi-lo)
    length=np.linalg.norm(d,axis=1)
    return track*length

R_cyl=1.0; zc=5.0; zmin=zc-0.1; zmax=zc+0.1; Vcyl=np.pi*R_cyl**2*0.2

def sample_source(N,rng):
    p=rng.uniform(-0.05,0.05,N); q=rng.uniform(-0.05,0.05,N)
    w=(1+p**2+q**2)**-1.5
    acc=rng.uniform(0,1,N)<w
    it=0
    while not acc.all():
        it+=1
        idx=np.where(~acc)[0]
        p2=rng.uniform(-0.05,0.05,len(idx)); q2=rng.uniform(-0.05,0.05,len(idx))
        w2=(1+p2**2+q2**2)**-1.5
        p[idx]=p2; q[idx]=q2
        acc[idx]=rng.uniform(0,1,len(idx))<w2
        if it>100: acc[:]=True
    n=np.sqrt(1+p*p+q*q)
    ux=p/n; uy=q/n; uz=1.0/n
    t=100.0/uz
    x0=ux*t; y0=uy*t; z0=np.zeros(N)
    return x0,y0,z0,ux,uy,uz

def photon_mc(N, rng=None, src_r=2.5, zsrc_lo=0.0, zsrc_hi=12.5, store_src=True, zc_cyl=5.0):
    if rng is None: rng=np.random.default_rng(12345)
    zminl=zc_cyl-0.1; zmaxl=zc_cyl+0.1
    E1=1.1732; E2=1.3325
    x0,y0,z0,ux,uy,uz=sample_source(N,rng)
    E=rng.choice([E1,E2],N)
    line0=np.column_stack([np.zeros(N),np.zeros(N),-100*np.ones(N)])
    dir0=np.column_stack([ux,uy,uz])
    far=line0+dir0*250.0
    L0=track_in_cylinder(line0,far,R_cyl,zminl,zmaxl)
    L0_sum=float(L0.sum())
    E=np.asarray(E,float); x=x0.copy(); y=y0.copy(); z=z0.copy()
    ux=ux.copy(); uy=uy.copy(); uz=uz.copy()
    prim=np.ones(N,int)
    alive=np.ones(N,bool)
    ph_p_tot=0.0; ph_s_tot=0.0; ph_p_E=0.0; ph_s_E=0.0
    spec_p=np.zeros(28); spec_s=np.zeros(28); ang_p=np.zeros(10); ang_s=np.zeros(10)
    L2p=np.zeros(4); L2s=np.zeros(4); ETF=np.zeros(4)
    L3_all=np.zeros(25); L3_prim=np.zeros(25); L3_dir=np.zeros(10)
    sumE_all=0.0; sumE_prim=0.0; nL3=0; Kdep=0.0
    srcE=[]; srcX=[]; srcY=[]; srcZ=[]; srcUx=[]; srcUy=[]; srcUz=[]; srcPrim=[]; srcPos=[]
    niter=0
    while alive.any():
        niter+=1
        n_alive=int(alive.sum())
        alidx=np.where(alive)[0]
        Ea=E[alidx]; xa=x[alidx]; ya=y[alidx]; za=z[alidx]; uxa=ux[alidx]; uya=uy[alidx]; uza=uz[alidx]; pra=prim[alidx]
        mu=mu_tot_w(Ea)
        s=-np.log(rng.uniform(0,1,n_alive)+1e-30)/mu
        r0=np.column_stack([xa,ya,za])
        r1=np.column_stack([xa+s*uxa, ya+s*uya, za+s*uza])
        # exit t to phantom boundary
        tz0=(0-za)/uza
        tz30=(30-za)/uza
        tx_lo=(-15-xa)/uxa; tx_hi=(15-xa)/uxa
        ty_lo=(-15-ya)/uya; ty_hi=(15-ya)/uya
        cand=np.stack([np.where(uxa>0,tx_hi,np.where(uxa<0,tx_lo,np.inf)),
                       np.where(uxa>0,tx_lo,np.where(uxa<0,tx_hi,np.inf)),
                       np.where(uya>0,ty_hi,np.where(uya<0,ty_lo,np.inf)),
                       np.where(uya>0,ty_lo,np.where(uya<0,ty_hi,np.inf)),
                       np.where(uza>0,tz30,np.where(uza<0,tz0,np.inf)),
                       np.where(uza>0,tz0,np.where(uza<0,tz30,np.inf))],axis=0)
        with np.errstate(invalid='ignore'):
            t_exit=np.min(cand,axis=0)
        t_exit=np.where(t_exit>0,t_exit,0.0)
        r1c=r1.copy()
        over=s>t_exit
        if over.any():
            r1c[over]=r0[over]+t_exit[over,None]*np.column_stack([uxa,uya,uza])[over]
        trk=track_in_cylinder(r0,r1c,R_cyl,zminl,zmaxl)
        has_trk=trk>0
        if has_trk.any():
            Eseg=Ea[has_trk]; cth=uza[has_trk]; pseg=pra[has_trk]; tseg=trk[has_trk]
            ph_p_tot+=float((tseg*(pseg==1)).sum()); ph_s_tot+=float((tseg*(pseg==0)).sum())
            ph_p_E+=float((tseg*Eseg*(pseg==1)).sum()); ph_s_E+=float((tseg*Eseg*(pseg==0)).sum())
            bi=np.clip(np.searchsorted(phE_bins,Eseg,side='right')-1,0,27)
            m1=pseg==1; m0=pseg==0
            if m1.any(): np.add.at(spec_p,bi[m1],tseg[m1])
            if m0.any(): np.add.at(spec_s,bi[m0],tseg[m0])
            cb=np.clip(np.searchsorted(ct_bins,cth,side='right')-1,0,9)
            if m1.any(): np.add.at(ang_p,cb[m1],tseg[m1])
            if m0.any(): np.add.at(ang_s,cb[m0],tseg[m0])
        inside=(r1c[:,2]>0)&(r1c[:,2]<30)&(np.abs(r1c[:,0])<15)&(np.abs(r1c[:,1])<15)
        x[alidx]=r1c[:,0]; y[alidx]=r1c[:,1]; z[alidx]=r1c[:,2]
        nxt=inside
        n_int=int(nxt.sum())
        if n_int>0:
            Eint=Ea[nxt]; xint=r1c[nxt,0]; yint=r1c[nxt,1]; zint=r1c[nxt,2]
            uxint=uxa[nxt]; uyint=uya[nxt]; uzint=uza[nxt]; prai=pra[nxt]
            frac_inc,frac_coh,frac_pho,frac_pai=mu_process(Eint)
            r=rng.uniform(0,1,n_int)
            typ=np.where(r<frac_inc,0,np.where(r<frac_inc+frac_coh,1,np.where(r<frac_inc+frac_coh+frac_pho,2,3)))
            incyl=(np.abs(zint-zc_cyl)<0.1)&(xint**2+yint**2<=R_cyl**2)
            insrc=(zint>=zsrc_lo)&(zint<=zsrc_hi)&(xint**2+yint**2<=src_r**2)
            for t0 in range(4):
                mm=(typ==t0)&incyl
                if mm.any():
                    pp=prai[mm]
                    L2p[t0]+=int((pp==1).sum()); L2s[t0]+=int((pp==0).sum())
            mC=typ==0
            if mC.any():
                Ei=Eint[mC]; ui=np.column_stack([uxint[mC],uyint[mC],uzint[mC]])
                gam=Ei/mec2
                mu=rng.uniform(-1,1,mC.sum())
                A=1+gam*(1-mu)
                f=(1.0/A**2)*(A+1.0/A-1.0+mu**2)
                acc=rng.uniform(0,1,mC.sum())<f/2.0
                it=0
                while not acc.all():
                    it+=1
                    idx=np.where(~acc)[0]; ng=len(idx)
                    mu2=rng.uniform(-1,1,ng); A2=1+gam[idx]*(1-mu2)
                    f2=(1.0/A2**2)*(A2+1.0/A2-1.0+mu2**2)
                    mu[idx]=mu2
                    acc[idx]=rng.uniform(0,1,ng)<f2/2.0
                    if it>50: acc[:]=True
                phi=rng.uniform(0,2*np.pi,mC.sum())
                Ep=Ei/(1+gam*(1-mu))
                Te=Ei-Ep
                e1,e2=basis(ui)
                sm=np.sqrt(np.maximum(0,1-mu**2))
                u_new=mu[:,None]*ui+sm[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
                pe=Ei[:,None]*ui-Ep[:,None]*u_new
                pe_n=np.linalg.norm(pe,axis=1)+1e-30
                ue=pe/pe_n[:,None]
                idx_glob=alidx[nxt][mC]
                E[idx_glob]=Ep; ux[idx_glob]=u_new[:,0]; uy[idx_glob]=u_new[:,1]; uz[idx_glob]=u_new[:,2]; prim[idx_glob]=0
                if store_src:
                    mm=insrc[mC]
                    if mm.any():
                        srcE.append(Te[mm]); srcX.append(xint[mC][mm]); srcY.append(yint[mC][mm]); srcZ.append(zint[mC][mm]); srcUx.append(ue[mm,0]); srcUy.append(ue[mm,1]); srcUz.append(ue[mm,2]); srcPrim.append(prai[mC][mm]); srcPos.append(np.zeros(mm.sum(),int))
                mm3=incyl[mC]
                if mm3.any():
                    Te3=Te[mm3]; pp3=prai[mC][mm3]; ue3=ue[mm3]
                    biel=np.clip(np.searchsorted(elE_bins,Te3,side='right')-1,0,24)
                    np.add.at(L3_all,biel,1.0)
                    mpp=pp3==1
                    if mpp.any(): np.add.at(L3_prim,biel[mpp],1.0)
                    cb3=np.clip(np.searchsorted(ct_bins,ue3[:,2],side='right')-1,0,9)
                    np.add.at(L3_dir,cb3,1.0)
                    sumE_all+=float(Te3.sum()); sumE_prim+=float(Te3[mpp].sum()); nL3+=len(Te3)
                    Kdep+=float(Te3.sum()); ETF[0]+=float(Te3.sum())
            mR=typ==1
            if mR.any():
                mui=1-0.02*rng.uniform(0,1,mR.sum())
                phi=rng.uniform(0,2*np.pi,mR.sum())
                ui=np.column_stack([uxint[mR],uyint[mR],uzint[mR]])
                e1,e2=basis(ui)
                sm=np.sqrt(np.maximum(0,1-mui**2))
                u_new=mui[:,None]*ui+sm[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
                idx_glob=alidx[nxt][mR]
                ux[idx_glob]=u_new[:,0]; uy[idx_glob]=u_new[:,1]; uz[idx_glob]=u_new[:,2]; prim[idx_glob]=0
            mP=typ==2
            if mP.any():
                Te=Eint[mP]
                idx_glob=alidx[nxt][mP]
                alive[idx_glob]=False
                mui=np.sqrt(rng.uniform(0,1,mP.sum()))
                phi=rng.uniform(0,2*np.pi,mP.sum())
                ui=np.column_stack([uxint[mP],uyint[mP],uzint[mP]])
                e1,e2=basis(ui)
                sm=np.sqrt(np.maximum(0,1-mui**2))
                ue=mui[:,None]*ui+sm[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
                if store_src:
                    mm=insrc[mP]
                    if mm.any():
                        srcE.append(Te[mm]); srcX.append(xint[mP][mm]); srcY.append(yint[mP][mm]); srcZ.append(zint[mP][mm]); srcUx.append(ue[mm,0]); srcUy.append(ue[mm,1]); srcUz.append(ue[mm,2]); srcPrim.append(prai[mP][mm]); srcPos.append(np.zeros(mm.sum(),int))
                mm3=incyl[mP]
                if mm3.any():
                    Te3=Te[mm3]; pp3=prai[mP][mm3]
                    biel=np.clip(np.searchsorted(elE_bins,Te3,side='right')-1,0,24)
                    np.add.at(L3_all,biel,1.0)
                    mpp=pp3==1
                    if mpp.any(): np.add.at(L3_prim,biel[mpp],1.0)
                    mui3=np.sqrt(rng.uniform(0,1,mm3.sum()))
                    # recompute ue for these (we have ue for all mP; subset)
                    ue3=ue[mm3]
                    cb3=np.clip(np.searchsorted(ct_bins,ue3[:,2],side='right')-1,0,9)
                    np.add.at(L3_dir,cb3,1.0)
                    sumE_all+=float(Te3.sum()); sumE_prim+=float(Te3[mpp].sum()); nL3+=mm3.sum()
                    Kdep+=float(Te3.sum()); ETF[2]+=float(Te3.sum())
            mQ=typ==3
            if mQ.any():
                U=Eint[mQ]-1.022
                Te=rng.uniform(0,1,mQ.sum())*U
                Tp=U-Te
                Te_all=np.concatenate([Te,Tp])
                idx_glob=alidx[nxt][mQ]
                alive[idx_glob]=False
                mui=1-0.1*rng.uniform(0,1,2*mQ.sum())
                phi=rng.uniform(0,2*np.pi,2*mQ.sum())
                ui=np.column_stack([uxint[mQ],uyint[mQ],uzint[mQ]])
                e1,e2=basis(ui)
                ui2=np.vstack([ui,ui]); e1b=np.vstack([e1,e1]); e2b=np.vstack([e2,e2])
                sm=np.sqrt(np.maximum(0,1-mui**2))
                ue=mui[:,None]*ui2+sm[:,None]*(np.cos(phi)[:,None]*e1b+np.sin(phi)[:,None]*e2b)
                pp_all=np.concatenate([prai[mQ],prai[mQ]])
                pos_all=np.concatenate([np.zeros(mQ.sum(),int),np.ones(mQ.sum(),int)])
                if store_src:
                    ins_all=np.concatenate([insrc[mQ],insrc[mQ]])
                    if ins_all.any():
                        srcE.append(Te_all[ins_all]); srcX.append(np.concatenate([xint[mQ],xint[mQ]])[ins_all]); srcY.append(np.concatenate([yint[mQ],yint[mQ]])[ins_all]); srcZ.append(np.concatenate([zint[mQ],zint[mQ]])[ins_all]); srcUx.append(ue[ins_all,0]); srcUy.append(ue[ins_all,1]); srcUz.append(ue[ins_all,2]); srcPrim.append(pp_all[ins_all]); srcPos.append(pos_all[ins_all])
                mm3=incyl[mQ]
                if mm3.any():
                    sel=np.concatenate([mm3,mm3])
                    Te3=Te_all[sel]; pp3=pp_all[sel]; ue3=ue[sel]
                    biel=np.clip(np.searchsorted(elE_bins,Te3,side='right')-1,0,24)
                    np.add.at(L3_all,biel,1.0)
                    mpp=pp3==1
                    if mpp.any(): np.add.at(L3_prim,biel[mpp],1.0)
                    cb3=np.clip(np.searchsorted(ct_bins,ue3[:,2],side='right')-1,0,9)
                    np.add.at(L3_dir,cb3,1.0)
                    sumE_all+=float(Te3.sum()); sumE_prim+=float(Te3[mpp].sum()); nL3+=len(Te3)
                    Kdep+=float(Te3.sum()); ETF[3]+=float(Te3.sum())
        dead=~inside
        alive[alidx[dead]]=False
        alidx2=np.where(alive)[0]
        lt=E[alidx2]<0.01
        alive[alidx2[lt]]=False
        if niter>200: alive[:]=False
    res=dict(L0_sum=L0_sum, ph_p_tot=ph_p_tot, ph_s_tot=ph_s_tot, ph_p_E=ph_p_E, ph_s_E=ph_s_E,
             spec_p=spec_p, spec_s=spec_s, ang_p=ang_p, ang_s=ang_s,
             L2p=L2p, L2s=L2s, ETF=ETF,
             L3_all=L3_all, L3_prim=L3_prim, L3_dir=L3_dir, sumE_all=sumE_all, sumE_prim=sumE_prim, nL3=nL3, Kdep=Kdep)
    if store_src:
        src=dict(E=np.concatenate(srcE),X=np.concatenate(srcX),Y=np.concatenate(srcY),Z=np.concatenate(srcZ),
                 Ux=np.concatenate(srcUx),Uy=np.concatenate(srcUy),Uz=np.concatenate(srcUz),
                 prim=np.concatenate(srcPrim).astype(int), pos=np.concatenate(srcPos).astype(int))
        res['src']=src
    res['niter']=niter
    return res
'''
open('mcbeam.py','w').write(code)
print('written')
