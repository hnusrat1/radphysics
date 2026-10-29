import numpy as np, math, time
mc2=0.5109989461; RHO=1.0
d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x=d[:,0]; mu_co=d[:,1]; mu_in=d[:,2]; mu_pe=d[:,3]; mu_pn=d[:,4]; mu_pef=d[:,5]; mu_tot=d[:,6]
def interp1(x,xx,yy): return np.interp(x,xx,yy)
def mu_total(E): return interp1(E,E_x,mu_tot)*RHO
def mu_coh(E):   return interp1(E,E_x,mu_co)*RHO
def mu_inc(E):   return interp1(E,E_x,mu_in)*RHO
def mu_pe_(E):   return interp1(E,E_x,mu_pe)*RHO
def mu_pair(E):  return (interp1(E,E_x,mu_pn)+interp1(E,E_x,mu_pef))*RHO
V_cyl=math.pi*0.2; PHI0=1/10.5**2
ZMINC,ZMAXC,RC=4.9,5.1,1.0
ph_edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
cos_edges=np.linspace(-1,1,11)
NB_PH=len(ph_edges)-1; NB_EL=len(el_edges)-1; NB_COS=len(cos_edges)-1
def ph_bin(E): return np.clip(np.searchsorted(ph_edges,E,side='right')-1,0,NB_PH-1)
def el_bin(E): return np.clip(np.searchsorted(el_edges,E,side='right')-1,0,NB_EL-1)
def cos_bin(c): return np.clip(np.searchsorted(cos_edges,c,side='right')-1,0,NB_COS-1)
def KN_f(alpha,cos):
    x=1+alpha*(1-cos); r=1/x; return r*r*(r+x-(1-cos*cos))
def sample_compton(E,rng):
    n=E.size; cos_s=np.zeros(n); Epr=np.zeros(n); cose=np.zeros(n); done=np.zeros(n,bool); idx=np.arange(n)
    while True:
        sub=idx[~done]
        if sub.size==0: break
        alpha=E[sub]/mc2; cs=rng.uniform(-1,1,sub.size)
        acc=rng.random(sub.size) < (KN_f(alpha,cs)/2.0)
        if acc.any():
            ai=sub[acc]; cos_s[ai]=cs[acc]; Ep=E[ai]/(1+alpha[acc]*(1-cs[acc])); Epr[ai]=Ep
            cose[ai]=(E[ai]-Ep*cs[acc])/np.sqrt(E[ai]**2+Ep**2-2*E[ai]*Ep*cs[acc]); done[ai]=True
    return cos_s,Epr,cose
def rotate_dir(d,cos,phi,rng):
    sin=np.sqrt(np.maximum(0,1-cos*cos)); sinp=np.sin(phi); cosp=np.cos(phi)
    kx,ky,kz=d[:,0],d[:,1],d[:,2]; nearz=np.abs(kz)>0.999
    nrm=np.sqrt(kx*kx+ky*ky); e1x=np.where(nearz,1.0,-ky/np.maximum(nrm,1e-12)); e1y=np.where(nearz,0.0,kx/np.maximum(nrm,1e-12)); e1z=np.zeros_like(kz)
    e1x[nearz]=1; e1y[nearz]=0
    e2x=ky*e1z-kz*e1y; e2y=kz*e1x-kx*e1z; e2z=kx*e1y-ky*e1x
    nx=cos*d[:,0]+sin*(cosp*e1x+sinp*e2x); ny=cos*d[:,1]+sin*(cosp*e1y+sinp*e2y); nz=cos*d[:,2]+sin*(cosp*e1z+sinp*e2z)
    out=np.stack([nx,ny,nz],1); return out/np.sqrt((out**2).sum(1))[:,None]
def cyl_track_vec(P0,D,L,zmin,zmax,rmax=1.0):
    n=P0.shape[0]; x0,y0,z0=P0[:,0],P0[:,1],P0[:,2]; dx,dy,dz=D[:,0],D[:,1],D[:,2]
    a=dx*dx+dy*dy; b=2*(x0*dx+y0*dy); c=x0*x0+y0*y0-rmax*rmax
    t_in=np.zeros(n); t_out=L.copy(); par=a<1e-14; npar=~par
    idx=np.where(npar)[0]
    if idx.size>0:
        aa=a[idx];bb=b[idx];cc=c[idx];LL=L[idx];disc=bb*bb-4*aa*cc;ok=disc>=0; ii=idx[ok]
        if ii.size>0:
            sq=np.sqrt(disc[ok]); t1=(-bb[ok]-sq)/(2*aa[ok]); t2=(-bb[ok]+sq)/(2*aa[ok]); tl=np.minimum(t1,t2); tu=np.maximum(t1,t2)
            t_in[ii]=np.maximum(0,tl); t_out[ii]=np.minimum(LL[ok],tu)
        t_in[idx[~ok]]=0; t_out[idx[~ok]]=0
    ip=np.where(par)[0]
    if ip.size>0:
        inside0=c[ip]<=0; t_in[ip[inside0]]=0; t_out[ip[inside0]]=L[ip[inside0]]; t_in[ip[~inside0]]=0; t_out[ip[~inside0]]=0
    adz=np.abs(dz); small=adz<1e-12; loz=np.zeros(n); hiz=L.copy()
    iz=np.where(~small)[0]
    if iz.size>0:
        z0i=z0[iz];dzi=dz[iz];Li=L[iz];t1=(zmin-z0i)/dzi;t2=(zmax-z0i)/dzi;tl=np.minimum(t1,t2);tu=np.maximum(t1,t2)
        loz[iz]=np.maximum(0,tl); hiz[iz]=np.minimum(Li,tu)
    ism=np.where(small)[0]
    if ism.size>0:
        inw=(z0[ism]>=zmin)&(z0[ism]<=zmax); t_out[ism[~inw]]=0
    lo=np.maximum(t_in,loz); hi=np.minimum(t_out,hiz); return np.maximum(0,hi-lo)

def run_batch(N, rng):
    flu_p=np.zeros((NB_PH,NB_COS)); flu_s=np.zeros((NB_PH,NB_COS))
    inter_all=np.zeros(4); inter_prim=np.zeros(4); inter_scat=np.zeros(4); trans=np.zeros(4)
    e_all=np.zeros(NB_EL); e_prim=np.zeros(NB_EL); e_cos=np.zeros(NB_COS)
    e_sum=0.0; e_sum_prim=0.0
    slab_src=np.zeros((150,NB_EL))
    E=np.where(rng.random(N)<0.5,1.1732,1.3325)
    xt=rng.uniform(-5,5,N); yt=rng.uniform(-5,5,N)
    pos=np.stack([xt,yt,np.zeros(N)],1)
    dirv=np.stack([xt,yt,np.full(N,100.0)],1); dirv/=np.sqrt((dirv**2).sum(1))[:,None]
    status=np.zeros(N,int); alive=np.ones(N,bool)
    while alive.any():
        idx=np.where(alive)[0]
        Ea=E[idx]; pa=pos[idx]; da=dirv[idx]; sa=status[idx]
        mut=mu_total(Ea)
        Lstep=-np.log(np.maximum(rng.random(Ea.size),1e-30))/mut
        tlen=cyl_track_vec(pa,da,Lstep,ZMINC,ZMAXC,RC)
        if tlen.sum()>0:
            eb=ph_bin(Ea); cb=cos_bin(da[:,2])
            m0=(sa==0)&(tlen>0)
            if m0.any(): np.add.at(flu_p,(eb[m0],cb[m0]),tlen[m0])
            m1=(sa==1)&(tlen>0)
            if m1.any(): np.add.at(flu_s,(eb[m1],cb[m1]),tlen[m1])
        tx1=(-15-pa[:,0])/da[:,0]; tx2=(15-pa[:,0])/da[:,0]
        ty1=(-15-pa[:,1])/da[:,1]; ty2=(15-pa[:,1])/da[:,1]
        tz1=(0-pa[:,2])/da[:,2];   tz2=(30-pa[:,2])/da[:,2]
        tmax=np.minimum(np.minimum(np.maximum(tx1,tx2),np.maximum(ty1,ty2)),np.maximum(tz1,tz2))
        exit_before = Lstep > (tmax+1e-9)
        interact = ~exit_before
        iE=Ea[interact]; ipos=pa[interact]; idir=da[interact]; ist=sa[interact]
        n_i=iE.size
        rproc=rng.random(n_i)*mu_total(iE)
        muc=mu_coh(iE); mui=mu_inc(iE); mup=mu_pe_(iE)
        proc=np.zeros(n_i,int)
        proc[(rproc>=muc)&(rproc<muc+mui)]=1
        proc[(rproc>=muc+mui)&(rproc<muc+mui+mup)]=2
        proc[rproc>=muc+mui+mup]=3
        incl = (ipos[:,2]>=ZMINC)&(ipos[:,2]<=ZMAXC)&(ipos[:,0]**2+ipos[:,1]**2<=RC**2)
        slab_idx = np.clip((ipos[:,2]/0.2).astype(int),0,149)
        if incl.any():
            pr=proc[incl]; st=ist[incl]; tr=np.zeros(incl.sum())
            comp_in=pr==1; pe_in=pr==2; pair_in=pr==3
            sel=incl&(proc==1)
            if sel.any():
                cs,Ep_,cse=sample_compton(iE[sel],rng)
                Te=iE[sel]-Ep_
                tr[comp_in]=Te
                eb=el_bin(Te); ceb=cos_bin(cse)
                np.add.at(e_all, eb, 1.0); np.add.at(e_cos, ceb, 1.0)
                e_sum += Te.sum()
                pm=ist[sel]==0
                if pm.any():
                    np.add.at(e_prim, eb[pm], 1.0); e_sum_prim += Te[pm].sum()
                np.add.at(slab_src, (slab_idx[sel], eb), 1.0)
            if pe_in.any():
                Te=iE[incl][pe_in]-0.0005
                tr[pe_in]=Te
                eb=el_bin(Te)
                np.add.at(e_all, eb, 1.0); np.add.at(e_cos, NB_COS-1, 1.0); e_sum += Te.sum()
                pm_=ist[incl][pe_in]==0
                if pm_.any(): np.add.at(e_prim, eb[pm_], 1.0); e_sum_prim += Te[pm_].sum()
                np.add.at(slab_src,(slab_idx[incl][pe_in], eb),1.0)
            if pair_in.any():
                Ee_=(iE[incl][pair_in]-1.022)/2.0
                tr[pair_in]=iE[incl][pair_in]-1.022
                eb=el_bin(Ee_)
                for k in range(2):
                    np.add.at(e_all, eb, 1.0); np.add.at(e_cos, NB_COS-1, 1.0); e_sum += Ee_.sum()
                    pm_=ist[incl][pair_in]==0
                    if pm_.any(): np.add.at(e_prim, eb[pm_],1.0); e_sum_prim += Ee_[pm_].sum()
                    np.add.at(slab_src,(slab_idx[incl][pair_in],eb),1.0)
            np.add.at(inter_all, pr, 1.0)
            m0=st==0
            if m0.any(): np.add.at(inter_prim, pr[m0], 1.0)
            m1=st==1
            if m1.any(): np.add.at(inter_scat, pr[m1], 1.0)
            np.add.at(trans, pr, tr)
        comp=proc==1
        if comp.any():
            cs,Ep,cse=sample_compton(iE[comp],rng)
            iE[comp]=Ep
            phi=rng.uniform(0,2*np.pi,comp.sum())
            idir[comp]=rotate_dir(idir[comp],cs,phi,rng)
            ist[comp]=1
        if (proc==0).any(): ist[proc==0]=1
        dead_proc=np.zeros(len(idx),bool)
        dead_proc[interact]=(proc==2)|(proc==3)
        Ea[interact]=iE; da[interact]=idir; sa[interact]=ist
        alive[idx]=False
        alive[idx[(~exit_before)&(~dead_proc)]]=True
    return flu_p,flu_s,inter_all,inter_prim,inter_scat,trans,e_all,e_prim,e_cos,e_sum,e_sum_prim,slab_src,N

rng=np.random.default_rng(12345)
t0=time.time(); res=run_batch(100000,rng); t1=time.time()
print('100k photons took', round(t1-t0,2), 's')
flu_p,flu_s,inter_a,inter_p,inter_s,trans,e_all,e_prim,e_cos,e_sum,e_sum_prim,slab_src,N=res
print('primary flu/Phi0 (expected ~0.7288):', flu_p.sum()/N/V_cyl/PHI0)
print('scattered flu/Phi0:', flu_s.sum()/N/V_cyl/PHI0)
print('total flu/Phi0:', (flu_p.sum()+flu_s.sum())/N/V_cyl/PHI0)
print('inter_all', inter_a, 'sum', inter_a.sum())
print('trans', trans)
print('e_all sum', e_all.sum(), 'e_prim sum', e_prim.sum())
print('mean init E all', e_sum/max(e_all.sum(),1), 'prim', e_sum_prim/max(e_prim.sum(),1))
print('e_cos', e_cos)
print('slab_src sum', slab_src.sum())