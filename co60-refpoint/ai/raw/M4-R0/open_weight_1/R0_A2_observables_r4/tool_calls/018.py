# Quick diagnostic: count process selections in the fixed MC
import numpy as np, time
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

# Check: at a batch of energies spanning the transport, what fractions?
Etest=np.array([1.3325,1.1732,1.05,1.023,0.7,0.5,0.2,0.1,0.05,0.02,1.28,1.1])
for E in Etest:
    mut=xs_lookup(TOT,np.array([E]))[0]
    pc=xs_lookup(COH,np.array([E]))[0]/mut
    pi_=xs_lookup(INC_i,np.array([E]))[0]/mut
    ppe=xs_lookup(PE,np.array([E]))[0]/mut
    pp=pair_xs(np.array([E]))[0]/mut
    print(f"E={E:.4f}: pc={pc:.2e} pinc={pi_:.4f} ppe={ppe:.2e} ppair={pp:.2e} sum={pc+pi_+ppe+pp:.5f}")

# Now a tiny run counting process selections
import sys
rng=np.random.default_rng(1)
N=20000
a=0.05; u=rng.uniform(-a,a,N); v=rng.uniform(-a,a,N)
E=np.where(rng.uniform(0,1,N)<0.5,1.1732,1.3325)
pos=np.stack([u*100.,v*100.,np.full(N,100.)],axis=1)
d=np.stack([u,v,np.ones(N)],axis=1); d/=np.sqrt(np.sum(d*d,axis=1,keepdims=True))
maxN=N+50000
P=np.zeros((maxN,3)); Dd=np.zeros((maxN,3)); En=np.zeros(maxN); Pr=np.zeros(maxN,bool); Al=np.zeros(maxN,bool)
P[:N]=pos; Dd[:N]=d; En[:N]=E; Pr[:N]=True; Al[:N]=True
nactive=N; free=list(range(N,maxN))
cnt={'ray':0,'inc':0,'pe':0,'pair':0,'esc':0}; pairE=[]
mec2=0.5109989461
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
it=0
while np.any(Al) and it<200:
    idx=np.nonzero(Al)[0]
    Ei=En[idx]; mut=xs_lookup(TOT,Ei)
    s_int=-np.log(np.maximum(rng.uniform(1e-300,1,len(idx)),1e-300))/np.maximum(mut,1e-6)
    p0=P[idx]; dd=Dd[idx]
    t1=(np.array([-15.,-15.,100.])-p0)/dd; t2=(np.array([15.,15.,130.])-p0)/dd
    tb=np.min(np.maximum(t1,t2),axis=1)
    s=np.minimum(s_int,tb); p1=p0+dd*s[:,None]
    inter=s_int<tb
    if inter.any():
        iidx=idx[inter]; Ei_i=Ei[inter]; pi=p1[inter]; di=dd[inter]; mut_i=mut[inter]
        pcoh=xs_lookup(COH,Ei_i)/mut_i; pinc=xs_lookup(INC_i,Ei_i)/mut_i; ppe=xs_lookup(PE,Ei_i)/mut_i; ppair=pair_xs(Ei_i)/mut_i
        rp=rng.uniform(0,1,len(iidx))
        isc=rp<pcoh; iin=(rp>=pcoh)&(rp<pcoh+pinc); ipe=(rp>=pcoh+pinc)&(rp<pcoh+pinc+ppe); ipair=rp>=pcoh+pinc+ppe
        cnt['ray']+=isc.sum(); cnt['inc']+=iin.sum(); cnt['pe']+=ipe.sum(); cnt['pair']+=ipair.sum()
        pairE.extend(list(Ei_i[ipair]))
        cont=isc|iin
        if cont.any():
            cidx=np.nonzero(cont)[0]; Econt=Ei_i[cont]; dcont=di[cont]; isinc=iin[cont]
            nE=Econt.copy(); c_all=np.empty(len(cidx))
            if isinc.sum()>0:
                ckn,Epkn=sample_KN(Econt[isinc],rng); c_all[isinc]=ckn; nE[isinc]=Epkn
            if (~isinc).sum()>0:
                nray=(~isinc).sum(); cR=np.empty(nray); got=np.zeros(nray,bool)
                while not got.all():
                    cc=rng.uniform(-1,1,nray)
                    acc=rng.uniform(0,1,nray)*2.0<(1+cc*cc)
                    new=(~got)&acc; cR[new]=cc[new]; got[new]=True
                c_all[~isinc]=cR
            sint=np.sqrt(np.maximum(0,1-c_all*c_all)); phi=rng.uniform(0,2*np.pi,len(cidx))
            dnew=rotate_vec(dcont,c_all,sint,np.cos(phi),np.sin(phi))
            Dd[iidx[cont]]=dnew; En[iidx[cont]]=nE; Pr[iidx[cont]]=False
        P[iidx]=pi; Al[iidx]=False
        if cont.any(): Al[iidx[cont]]=True
        if ipair.any():
            for sgn in (1,-1):
                if not free:
                    ext=10000; P=np.vstack([P,np.zeros((ext,3))]); Dd=np.vstack([Dd,np.zeros((ext,3))])
                    En=np.concatenate([En,np.zeros(ext)]); Pr=np.concatenate([Pr,np.zeros(ext,bool)]); Al=np.concatenate([Al,np.zeros(ext,bool)])
                    free.extend(range(maxN,maxN+ext)); maxN+=ext
                fi=free.pop(); P[fi]=pi[0 if False else 0]; Dd[fi]=np.array([1.,0.,0.]); En[fi]=0.511; Pr[fi]=False; Al[fi]=True
    cnt['esc']+=(~inter).sum()
    Al[idx[~inter]]=False
    it+=1
print("counts:",cnt,"total interactions:",cnt['ray']+cnt['inc']+cnt['pe']+cnt['pair'])
print("pair E histogram:",np.histogram(np.array(pairE),bins=[0.5,1.02,1.1,1.2,1.3,1.4])[0] if pairE else None)
print("pair energies:",np.sort(np.array(pairE))[:20] if pairE else None)