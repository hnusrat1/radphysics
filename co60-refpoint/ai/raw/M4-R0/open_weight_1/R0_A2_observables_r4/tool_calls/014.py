# Debug slab_sec[0]: which photons contribute
import numpy as np, time
from scipy.interpolate import interp1d
# --- reuse setup by importing from the previous cell is not possible; re-code minimal debug ---
ph = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
Eph=ph[:,0]; coh=ph[:,1]; inc=ph[:,2]; pe=ph[:,3]; pn=ph[:,4]; pel=ph[:,5]; tot=ph[:,6]
Elg=np.logspace(np.log10(0.01),np.log10(2.0),2500)
def interp_log(x,y,xq):
    ok=(y>0)&(x>0); fi=interp1d(np.log(x[ok]),np.log(y[ok]),kind='linear',bounds_error=False,fill_value='extrapolate')
    return np.exp(fi(np.log(xq)))
COH=interp_log(Eph,coh,Elg); INC_i=interp_log(Eph,inc,Elg); PE=interp_log(Eph,pe,Elg)
PN=interp_log(Eph,pn,Elg); PEL=interp_log(Eph,pel,Elg); TOT=interp_log(Eph,tot,Elg)
logElg=np.log(Elg)
def xs_lookup(arr,E): return np.exp(np.interp(np.log(np.maximum(E,2e-3)),logElg,np.log(arr)))
mec2=0.5109989461; E1=1.1732; E2=1.3325
pe_edges=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])

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
    a=E/mec2; Ep=E/(1+a*(1-c)); r=Ep/E
    return r*r*(r+1/r-(1-c*c))
def sample_KN(Earr,rng):
    n=len(Earr); c=np.empty(n); got=np.zeros(n,bool); fmax=KN_weight(-np.ones(n),Earr)
    while not got.all():
        cc=rng.uniform(-1,1,n)
        acc=(rng.uniform(0,1,n)*fmax)<KN_weight(cc,Earr)
        new=(~got)&acc; c[new]=cc[new]; got[new]=True
    a=Earr/mec2; Ep=Earr/(1+a*(1-c)); return c,Ep

# small debug run: N=20000, track contributions to slab_sec[0] with origin info
rng=np.random.default_rng(3)
N=20000
d0,u,v=sample_source_dir(N,rng)
E=np.where(rng.uniform(0,1,N)<0.5,E1,E2)
pos=np.stack([u*100.,v*100.,np.full(N,100.)],axis=1)
maxN=N+50000
P=np.zeros((maxN,3)); Dd=np.zeros((maxN,3)); En=np.zeros(maxN); Pr=np.zeros(maxN,bool); Al=np.zeros(maxN,bool)
P[:N]=pos; Dd[:N]=d0; En[:N]=E; Pr[:N]=True; Al[:N]=True
nactive=N; free=list(range(N,maxN))
# record: for slab_sec[0] accumulate (E, prim, chord)
dbg=[]
it=0
while np.any(Al):
    idx=np.nonzero(Al)[0]
    Ei=En[idx]; mut=xs_lookup(TOT,Ei)
    s_int=-np.log(np.maximum(rng.uniform(1e-300,1,len(idx)),1e-300))/np.maximum(mut,1e-6)
    p0=P[idx]; dd=Dd[idx]
    t1=(np.array([-15.,-15.,100.])-p0)/dd; t2=(np.array([15.,15.,130.])-p0)/dd
    tlo=np.maximum(np.minimum(t1,t2),0); thi=np.maximum(t1,t2); tb=np.min(thi,axis=1)
    s=np.minimum(s_int,tb); p1=p0+dd*s[:,None]
    chs=cyl_chord(p0,p1,1.0,100.125,0.125)  # slab 0
    sm=chs>0
    if sm.any():
        for (Ee_,pp_,cc_) in zip(Ei[sm],Pr[idx][sm],chs[sm]):
            dbg.append((Ee_,pp_,cc_))
    inter=s_int<tb
    if inter.any():
        iidx=idx[inter]; Ei_i=Ei[inter]; pi=p1[inter]; di=dd[inter]; mut_i=mut[inter]
        pc=xs_lookup(COH,Ei_i)/mut_i; pinc=xs_lookup(INC_i,Ei_i)/mut_i; ppe=xs_lookup(PE,Ei_i)/mut_i
        rp=rng.uniform(0,1,len(iidx))
        isc=rp<pc; iin=(rp>=pc)&(rp<pc+pinc); ipe=(rp>=pc+pinc)&(rp<pc+pinc+ppe); ipair=rp>=pc+pinc+ppe
        cont=isc|iin
        if cont.any():
            cidx=np.nonzero(cont)[0]; Econt=Ei_i[cont]; dcont=di[cont]; isinc=iin[cont]
            nE=Econt.copy(); c_all=np.empty(len(cidx))
            if np.sum(isinc)>0:
                ckn,Epkn=sample_KN(Econt[isinc],rng); c_all[isinc]=ckn; nE[isinc]=Epkn
            if np.sum(~isinc)>0:
                nray=np.sum(~isinc); cR=np.empty(nray); got=np.zeros(nray,bool)
                while not got.all():
                    cc=rng.uniform(-1,1,nray)
                    acc=rng.uniform(0,1,nray)*2.0<(1+cc*cc)
                    new=(~got)&acc; cR[new]=cc[new]; got[new]=True
                c_all[~isinc]=cR
            sint=np.sqrt(np.maximum(0,1-c_all*c_all)); phi=rng.uniform(0,2*np.pi,len(cidx))
            dnew=rotate_vec(dcont,c_all,sint,np.cos(phi),np.sin(phi))
            Dd[iidx[cont]]=dnew; En[iidx[cont]]=nE; Pr[iidx[cont]]=False
        Al[iidx]=False
        if cont.any(): Al[iidx[cont]]=True
        if ipair.any():
            pidx=np.nonzero(ipair)[0]
            for q in pidx:
                u2=rng.normal(size=3); u2/=np.linalg.norm(u2); u2-=di[q]*np.dot(u2,di[q])
                if np.linalg.norm(u2)<1e-9: u2=np.array([1.,0.,0.])
                u2/=np.linalg.norm(u2)
                if len(free)<2:
                    ext=10000; P=np.vstack([P,np.zeros((ext,3))]); Dd=np.vstack([Dd,np.zeros((ext,3))])
                    En=np.concatenate([En,np.zeros(ext)]); Pr=np.concatenate([Pr,np.zeros(ext,bool)]); Al=np.concatenate([Al,np.zeros(ext,bool)])
                    free.extend(range(maxN,maxN+ext)); maxN+=ext
                for sgn in (1,-1):
                    fi=free.pop(); P[fi]=pi[q]; Dd[fi]=sgn*u2; En[fi]=0.511; Pr[fi]=False; Al[fi]=True
                    if fi>=nactive: nactive=fi+1
    Al[idx[~inter]]=False
    it+=1

dbg=np.array(dbg)
print("dbg entries:",len(dbg))
prim_dbg=dbg[dbg[:,1]==1]; sec_dbg=dbg[dbg[:,1]==0]
print("primary entries:",len(prim_dbg),"track sum:",prim_dbg[:,2].sum())
print("scattered entries:",len(sec_dbg),"track sum:",sec_dbg[:,2].sum())
print("primary energy hist:",np.histogram(prim_dbg[:,0],bins=pe_edges)[0])
print("scattered energy hist:",np.histogram(sec_dbg[:,0],bins=pe_edges)[0])
print("scattered energies sorted sample:",np.sort(sec_dbg[:,0])[-20:])