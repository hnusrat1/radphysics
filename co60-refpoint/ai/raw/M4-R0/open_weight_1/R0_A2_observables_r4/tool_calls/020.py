# Final photon MC with corrected process sampling (relative weights), N=2.5e6
import numpy as np, time, json
from scipy.interpolate import interp1d

ph = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
Eph=ph[:,0]; coh=ph[:,1]; inc=ph[:,2]; pe=ph[:,3]; pn=ph[:,4]; pel=ph[:,5]; tot=ph[:,6]
Elg=np.logspace(np.log10(0.01),np.log10(2.0),2500)
def interp_log(x,y,xq):
    ok=(y>0)&(x>0); fi=interp1d(np.log(x[ok]),np.log(y[ok]),kind='linear',bounds_error=False,fill_value='extrapolate')
    return np.exp(fi(np.log(xq)))
COH=interp_log(Eph,coh,Elg); INC_i=interp_log(Eph,inc,Elg); PE=interp_log(Eph,pe,Elg)
PN0=interp_log(Eph,pn,Elg); PEL0=interp_log(Eph,pel,Elg); TOT=interp_log(Eph,tot,Elg)
logElg=np.log(Elg)
def xs_lookup(arr,E): return np.exp(np.interp(np.log(np.maximum(E,2e-3)),logElg,np.log(arr)))
def pair_xs(E):
    E=np.maximum(E,0)
    return np.where(E>1.022, xs_lookup(PN0,E)+xs_lookup(PEL0,E), 0.0)
el=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
Ee=el[:,0]; Y=el[:,5]
def Yrad(E): return float(np.interp(E,Ee,Y))
mec2=0.5109989461; E1=1.1732; E2=1.3325
pe_edges=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ct_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
nPE=len(pe_edges)-1; nCT=len(ct_edges)-1

def cyl_chord(p0,p1,R,zc,hh):
    d=p1-p0; L=np.sqrt(np.sum(d*d,axis=1))
    dx=d[:,0]; dy=d[:,1]; dz=d[:,2]; x0=p0[:,0]; y0=p0[:,1]; z0=p0[:,2]
    a=dx*dx+dy*dy; b=x0*dx+y0*dy; c=x0*x0+y0*y0-R*R
    big=a>1e-16
    sq=np.sqrt(np.maximum(b*b-a*c,0))
    t1=(-b-sq)/np.maximum(a,1e-30); t2=(-b+sq)/np.maximum(a,1e-30)
    lo=np.minimum(t1,t2); hi=np.maximum(t1,t2)
    lo[~big]=-1e9; hi[~big]=1e9
    outside=(~big)&(x0*x0+y0*y0>=R*R)
    invdz=np.where(np.abs(dz)>1e-12,1.0/dz,0.0)
    z1=(zc-hh-z0)*invdz; z2=(zc+hh-z0)*invdz
    zlo=np.minimum(z1,z2); zhi=np.maximum(z1,z2)
    dz0=np.abs(dz)<1e-12
    zlo[dz0]=0.; zhi[dz0]=1.
    bad=dz0&(np.abs(z0-zc)>hh); zlo[bad]=1.; zhi[bad]=0.
    lo=np.maximum(np.maximum(lo,zlo),0); hi=np.minimum(np.minimum(hi,zhi),1)
    chord=L*np.maximum(0,hi-lo); chord[outside]=0
    return chord
def sample_source_dir(n,rng):
    a=0.05; maxw=(1+2*a*a)**1.5
    u=np.empty(n); v=np.empty(n); got=np.zeros(n,bool)
    while not got.all():
        uu=rng.uniform(-a,a,n); vv=rng.uniform(-a,a,n)
        acc=rng.uniform(0,1,n)*maxw < (1+uu*uu+vv*vv)**1.5
        new=(~got)&acc; u[new]=uu[new]; v[new]=vv[new]; got[new]=True
    d=np.stack([u,v,np.ones_like(u)],axis=1); d/=np.sqrt(np.sum(d*d,axis=1,keepdims=True))
    return d,u,v
def rotate_vec(d,cost,sint,cosphi,sinphi):
    aux=np.broadcast_to(np.array([0.,0.,1.]),(len(d),3)).copy()
    aux[np.abs(d[:,2])>0.9]=np.array([1.,0.,0.])
    e1=np.cross(aux,d); e1=e1/np.maximum(np.sqrt(np.sum(e1*e1,axis=1,keepdims=True)),1e-12)
    e2=np.cross(d,e1)
    return cost[:,None]*d+sint[:,None]*(cosphi[:,None]*e1+sinphi[:,None]*e2)
def KN_weight(c,E):
    a=E/mec2; Ep=E/(1+a*(1-c)); rr=Ep/E
    return rr*rr*(rr+1/rr-(1-c*c))
def sample_KN(Earr,rng):
    n=len(Earr); c=np.empty(n); got=np.zeros(n,bool); fmax=KN_weight(-np.ones(n),Earr)
    while not got.all():
        cc=rng.uniform(-1,1,n)
        acc=(rng.uniform(0,1,n)*fmax)<KN_weight(cc,Earr)
        new=(~got)&acc; c[new]=cc[new]; got[new]=True
    a=Earr/mec2; Ep=Earr/(1+a*(1-c)); return c,Ep

Egrid=np.logspace(np.log10(0.01),np.log10(1.5),300)
from scipy.integrate import quad
TR=np.zeros_like(Egrid); TRY=np.zeros_like(Egrid)
for j,Eg in enumerate(Egrid):
    Z=quad(lambda c: KN_weight(c,Eg),-1,1)[0]
    TR[j]=quad(lambda c: KN_weight(c,Eg)*(Eg-Eg/(1+(Eg/mec2)*(1-c))),-1,1)[0]/Z
    TRY[j]=quad(lambda c: KN_weight(c,Eg)*(Eg-Eg/(1+(Eg/mec2)*(1-c)))*(1-Yrad(Eg-Eg/(1+(Eg/mec2)*(1-c)))),-1,1)[0]/Z
logEg=np.log(Egrid)
def TR_arr(Ea): return np.exp(np.interp(np.log(np.maximum(Ea,0.01)),logEg,np.log(np.maximum(TR,1e-30))))
def TRY_arr(Ea): return np.exp(np.interp(np.log(np.maximum(Ea,0.01)),logEg,np.log(np.maximum(TRY,1e-30))))
def muenE_arr(Ea):
    pev=xs_lookup(PE,Ea); incv=xs_lookup(INC_i,Ea); pnv=pair_xs(Ea)
    val=pev*(Ea-0.0005)*(1-np.interp(np.maximum(Ea-0.0005,0.01),Ee,Y))+incv*TRY_arr(Ea)
    Te=(Ea-1.022)/2.0
    val+=(Ea>1.022)*pnv*2*Te*(1-np.interp(np.maximum(Te,0.01),Ee,Y))
    return val

def run_photon_mc(N,seed=1,slab_dz=0.25,zmax=8.0,maxit=3000000):
    rng=np.random.default_rng(seed)
    d0,u,v=sample_source_dir(N,rng)
    E=np.where(rng.uniform(0,1,N)<0.5,E1,E2)
    pos=np.stack([u*100.,v*100.,np.full(N,100.)],axis=1)
    maxN=N+300000
    P=np.zeros((maxN,3)); Dd=np.zeros((maxN,3)); En=np.zeros(maxN)
    Pr=np.zeros(maxN,bool); Al=np.zeros(maxN,bool)
    P[:N]=pos; Dd[:N]=d0; En[:N]=E; Pr[:N]=True; Al[:N]=True
    nactive=N; free_idx=list(range(N,maxN))
    flu_prim=np.zeros(nPE); flu_sec=np.zeros(nPE); ang_prim=np.zeros(nCT); ang_sec=np.zeros(nCT)
    Eflu_prim=0.0; Eflu_sec=0.0
    nz=int(zmax/slab_dz); z_edges=np.linspace(0,zmax,nz+1); zc_slabs=0.5*(z_edges[:-1]+z_edges[1:]); hh_slab=0.5*slab_dz
    slab_prim=np.zeros((2,nz)); slab_sec=np.zeros((nz,nPE)); slab_Kcol=np.zeros(nz)
    t0=time.time(); it=0; pair_spawn=0
    while np.any(Al):
        idx=np.nonzero(Al)[0]
        if len(idx)==0: break
        Ei=En[idx]; mut=xs_lookup(TOT,Ei)
        s_int=-np.log(np.maximum(rng.uniform(1e-300,1,len(idx)),1e-300))/np.maximum(mut,1e-6)
        p0=P[idx]; dd=Dd[idx]
        t1=(np.array([-15.,-15.,100.])-p0)/dd; t2=(np.array([15.,15.,130.])-p0)/dd
        tlo=np.maximum(np.minimum(t1,t2),0); thi=np.maximum(t1,t2); tb=np.min(thi,axis=1)
        s=np.minimum(s_int,tb); p1=p0+dd*s[:,None]
        ch=cyl_chord(p0,p1,1.0,105.0,0.1)
        eb=np.searchsorted(pe_edges,Ei,side='right')-1
        cb=np.searchsorted(ct_edges,dd[:,2],side='right')-1
        isprim=Pr[idx]
        m=ch>0
        if m.any():
            chm=ch[m]; eb_m=eb[m]; cb_m=cb[m]; pm=isprim[m]; Em=Ei[m]
            for i in range(nPE):
                mm=eb_m==i
                if mm.any():
                    chmm=chm[mm]; pmm=pm[mm]
                    flu_prim[i]+=np.sum(chmm[pmm]); flu_sec[i]+=np.sum(chmm[~pmm])
            for j in range(nCT):
                mm=cb_m==j
                if mm.any():
                    chmm=chm[mm]; pmm=pm[mm]
                    ang_prim[j]+=np.sum(chmm[pmm]); ang_sec[j]+=np.sum(chmm[~pmm])
            Eflu_prim+=np.sum(chm[pm]*Em[pm]); Eflu_sec+=np.sum(chm[~pm]*Em[~pm])
        for k in range(nz):
            chs=cyl_chord(p0,p1,1.0,100.0+zc_slabs[k],hh_slab)
            sm=chs>0
            if sm.any():
                es=Ei[sm]; ps=isprim[sm]; chsm=chs[sm]
                eb_s=np.searchsorted(pe_edges,es,side='right')-1
                for i in range(nPE):
                    mm=eb_s==i
                    if mm.any():
                        mm2=~ps[mm]
                        if mm2.any():
                            slab_sec[k,i]+=np.sum(chsm[mm][mm2])
                slab_prim[0,k]+=np.sum(chsm[ps&(es==E1)])
                slab_prim[1,k]+=np.sum(chsm[ps&(es==E2)])
            slab_Kcol[k]+=np.sum(chs*muenE_arr(Ei))
        inter=s_int<tb
        if inter.any():
            iidx=idx[inter]; Ei_i=Ei[inter]; pi=p1[inter]; di=dd[inter]; mut_i=mut[inter]
            ni=len(iidx)
            wcoh=xs_lookup(COH,Ei_i); winc=xs_lookup(INC_i,Ei_i); wpe=xs_lookup(PE,Ei_i); wpair=pair_xs(Ei_i)
            W=wcoh+winc+wpe+wpair
            pcoh=wcoh/W; pinc=winc/W; ppe=wpe/W; ppair=wpair/W
            rp=rng.uniform(0,1,ni)
            isc=rp<pcoh; iin=(rp>=pcoh)&(rp<pcoh+pinc); ipe=(rp>=pcoh+pinc)&(rp<pcoh+pinc+ppe); ipair=rp>=pcoh+pinc+ppe
            cont=isc|iin
            if cont.any():
                cidx=np.nonzero(cont)[0]
                Econt=Ei_i[cont]; dcont=di[cont]; isinc=iin[cont]
                nE=Econt.copy(); c_all=np.empty(len(cidx))
                ninc=np.sum(isinc)
                if ninc>0:
                    ckn,Epkn=sample_KN(Econt[isinc],rng); c_all[isinc]=ckn; nE[isinc]=Epkn
                nray=np.sum(~isinc)
                if nray>0:
                    cR=np.empty(nray); got=np.zeros(nray,bool)
                    while not got.all():
                        cc=rng.uniform(-1,1,nray)
                        acc=rng.uniform(0,1,nray)*2.0<(1+cc*cc)
                        new=(~got)&acc; cR[new]=cc[new]; got[new]=True
                    c_all[~isinc]=cR
                sint=np.sqrt(np.maximum(0,1-c_all*c_all))
                phi=rng.uniform(0,2*np.pi,len(cidx))
                dnew=rotate_vec(dcont,c_all,sint,np.cos(phi),np.sin(phi))
                Dd[iidx[cont]]=dnew; En[iidx[cont]]=nE; Pr[iidx[cont]]=False
            P[iidx]=pi
            Al[iidx]=False
            if cont.any(): Al[iidx[cont]]=True
            if ipair.any():
                pidx=np.nonzero(ipair)[0]
                for q in pidx:
                    u2=rng.normal(size=3); u2/=np.linalg.norm(u2)
                    u2-=di[q]*np.dot(u2,di[q])
                    if np.linalg.norm(u2)<1e-9: u2=np.array([1.,0.,0.])
                    u2/=np.linalg.norm(u2)
                    if len(free_idx)<2:
                        ext=10000
                        P=np.vstack([P,np.zeros((ext,3))]); Dd=np.vstack([Dd,np.zeros((ext,3))])
                        En=np.concatenate([En,np.zeros(ext)]); Pr=np.concatenate([Pr,np.zeros(ext,bool)]); Al=np.concatenate([Al,np.zeros(ext,bool)])
                        free_idx.extend(range(maxN,maxN+ext)); maxN+=ext
                    for sgn in (1,-1):
                        fi=free_idx.pop()
                        P[fi]=pi[q]; Dd[fi]=sgn*u2; En[fi]=0.511; Pr[fi]=False; Al[fi]=True
                        if fi>=nactive: nactive=fi+1
                        pair_spawn+=1
        escaped=~inter
        Al[idx[escaped]]=False
        it+=1
        if it>maxit: break
    return dict(N=N,flu_prim=flu_prim,flu_sec=flu_sec,ang_prim=ang_prim,ang_sec=ang_sec,
                Eflu_prim=Eflu_prim,Eflu_sec=Eflu_sec,slab_prim=slab_prim,slab_sec=slab_sec,
                slab_Kcol=slab_Kcol,zc_slabs=zc_slabs,hh_slab=hh_slab,pair_spawn=pair_spawn,
                time=time.time()-t0,it=it)

if __name__=='__main__':
    r=run_photon_mc(2500000,seed=2024)
    print("time %.1f it %d pair %d"%(r['time'],r['it'],r['pair_spawn']))
    print("flu_prim/N:",r['flu_prim'].sum()/r['N'])
    print("flu_sec/N:",r['flu_sec'].sum()/r['N'])
    print("Eflu_prim/N:",r['Eflu_prim']/r['N'],"Eflu_sec/N:",r['Eflu_sec']/r['N'])
    np.save('/tmp/photon_mc_final.npy',r,allow_pickle=True)
    print("saved")