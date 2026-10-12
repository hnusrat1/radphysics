
code = r'''
import numpy as np, sys, json
seed=int(sys.argv[1]); N_hist=int(sys.argv[2])
rng=np.random.default_rng(seed)
re=2.8179403e-13; me=0.510998950
NA=6.02214076e23
Ne=0.5551*NA            # electrons per cm3 of water (rho=1)

def sigKN(E):
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)-(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2)
_pe_E=np.array([1.0,1.022,1.5,2.0,3.0,4.0,6.0,10.0,20.0,100.0])
_pe_s=np.array([0,0,0.00005,0.00035,0.0025,0.0055,0.0120,0.0200,0.0216,0.0216])
def sig_pair(E): return np.interp(E,_pe_E,_pe_s)          # cm2/g (approx.)
def sig_coh(E):  return 0.00014*(2.0/np.maximum(E,1e-3))**1.2
def sig_pe(E):   return 0.0028*(0.1/np.maximum(E,1e-4))**3

def sample_compton(E):
    # proposal in eps ~ 1/eps (log-uniform); f/(1/eps)=1+eps^2-eps sin^2 <= 2
    n=E.size; k=E/me
    epsmin=1/(1+2*k); L=np.log(1/epsmin)
    out=np.empty(n); todo=np.arange(n)
    while todo.size:
        kk=k[todo]
        eps=np.exp(-L[todo]*rng.random(todo.size))
        c=1-(1/eps-1)/kk
        ratio=(1+eps*eps-eps*(1-c*c))/2.0
        acc=rng.random(todo.size)<ratio
        out[todo[acc]]=eps[acc]
        todo=todo[~acc]
    return out

def unit(d): return d/np.linalg.norm(d,axis=1,keepdims=True)
def perp_basis(d):
    a=np.where(np.abs(d[:,[0]])<0.9,np.array([[1.0,0,0]]),np.array([[0,1.0,0]]))
    u=unit(np.cross(d,a)); v=np.cross(d,u)
    return u,v
def rotate(d,cth,phi):
    u,v=perp_basis(d)
    s=np.sqrt(np.maximum(0,1-cth*cth))
    return unit(cth[:,None]*d+s[:,None]*(np.cos(phi)[:,None]*u+np.sin(phi)[:,None]*v))
def iso_dir(n):
    c=2*rng.random(n)-1; ph=2*np.pi*rng.random(n); s=np.sqrt(1-c*c)
    return np.stack([s*np.cos(ph),s*np.sin(ph),c],axis=1)

def sample_rayleigh_c(E):
    # pdf(c) ~ (1+c^2) exp(-A(1-c)); A=2k^2/q0^2 (k in 1/A, q0=2.5/A) -- crude form-factor model
    kA=E*1000.0/12.398; A=2*kA**2/2.5**2
    n=E.size; out=np.empty(n); todo=np.arange(n)
    while todo.size:
        u=rng.exponential(1.0,todo.size)/A[todo]
        ok=u<=2.0
        acc=ok & (rng.random(todo.size)*2.0 < (1+(1-np.minimum(u,2))**2))
        out[todo[acc]]=1-u[acc]
        todo=todo[~acc]
    return out

def sample_pe_cos(E):
    # photoelectron direction, pdf ~ sin^2/(1-beta cos)^4 ; bound by sin^2 ... use rejection with bound 1/(1-beta)^2
    b=np.sqrt(1-(me/(E+me))**2)
    n=E.size; out=np.empty(n); todo=np.arange(n)
    while todo.size:
        c=2*rng.random(todo.size)-1; bb=b[todo]
        f=(1-c*c)/(1-bb*c)**4
        bound=1.0/(1-bb)**2*0.5+1e-300
        bound=np.maximum(bound, f.max())  # safe global bound per batch
        acc=rng.random(todo.size)*bound<f
        out[todo[acc]]=c[acc]
        todo=todo[~acc]
    return out

LO=np.array([-15.0,-15.0,0.0]); HI=np.array([15.0,15.0,30.0])
def in_volume(p): return (np.hypot(p[:,0],p[:,1])<=1.0)&(np.abs(p[:,2]-10.0)<=0.1)
def dist_to_exit(p,d):
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(LO-p)/d; t2=(HI-p)/d
    tmax=np.where(d>0,t2,np.where(d<0,t1,np.inf))
    return np.min(tmax,axis=1)

# ---- source ----
u=rng.uniform(-0.05,0.05,N_hist); v=rng.uniform(-0.05,0.05,N_hist)
acc=rng.random(N_hist)<(1+u*u+v*v)**(-1.5)      # uniform in solid angle
u=u[acc]; v=v[acc]
dr=unit(np.stack([u,v,np.ones_like(u)],axis=1))
pos=np.stack([100*u,100*v,np.zeros_like(u)],axis=1)   # entrance face z=0 (source at z=-100)
en=np.full(pos.shape[0],2.0)

rows=[]
it=0
while pos.shape[0]>0 and it<500:
    it+=1
    comp=sigKN(en)*Ne; pe=sig_pe(en); pair=sig_pair(en)*1.0; coh=sig_coh(en)
    tot=comp+pe+pair+coh
    s=-np.log(rng.random(pos.shape[0]))/tot
    tex=dist_to_exit(pos,dr)
    alive=s<tex
    pos,dr,en,comp,pe,pair,coh,tot,s=[x[alive] for x in (pos,dr,en,comp,pe,pair,coh,tot,s)]
    if pos.shape[0]==0: break
    pos=pos+s[:,None]*dr
    vol=in_volume(pos)
    r=rng.random(pos.shape[0])*tot
    pm=np.zeros(pos.shape[0],int)
    pm[r>=comp]=1; pm[r>=comp+pe]=2; pm[r>=comp+pe+pair]=3
    # pm: 0 compton, 1 photo, 2 pair, 3 coherent
    newE=en.copy(); newd=dr.copy(); keep=np.ones(pos.shape[0],bool)
    ann_pos=[]; ann_d=[]; ann_E=[]
    # --- Compton
    m=pm==0
    if m.any():
        Ei=en[m]; eps=sample_compton(Ei); Eo=Ei*eps; k=Ei/me
        cth=1-(1/eps-1)/k; phi=2*np.pi*rng.random(Ei.size)
        din=dr[m]
        newd_c=rotate(din,cth,phi)
        pvec=Ei[:,None]*din-Eo[:,None]*newd_c
        pmag=np.linalg.norm(pvec,axis=1)
        cos_e=np.clip((Ei-Eo*cth)/np.maximum(pmag,1e-12),-1,1)
        Te=Ei-Eo
        newd[m]=newd_c; newE[m]=Eo
        mi=np.where(vol&m)[0]
        for j in mi:
            q=np.searchsorted(np.where(m)[0],j)
            rows.append(dict(process="compton",e_in_MeV=float(en[j]),e_out_MeV=float(Eo[q]),cos_scatter=float(cth[q]),
                             t_e_MeV=float(Te[q]),cos_electron=float(cos_e[q]),t_positron_MeV=None))
    # --- coherent
    m3=pm==3
    if m3.any():
        Ei=en[m3]; c3=sample_rayleigh_c(Ei); ph=2*np.pi*rng.random(Ei.size)
        newd[m3]=rotate(dr[m3],c3,ph)
        mi=np.where(vol&m3)[0]
        for j in mi:
            q=np.searchsorted(np.where(m3)[0],j)
            rows.append(dict(process="rayleigh",e_in_MeV=float(en[j]),e_out_MeV=float(en[j]),cos_scatter=float(c3[q]),
                             t_e_MeV=0.0,cos_electron=None,t_positron_MeV=None))
    # --- photoelectric
    m1=pm==1
    if m1.any():
        Ei=en[m1]; cp=sample_pe_cos(Ei)
        mi=np.where(vol&m1)[0]
        for j in mi:
            q=np.searchsorted(np.where(m1)[0],j)
            rows.append(dict(process="photoelectric",e_in_MeV=float(en[j]),e_out_MeV=None,cos_scatter=None,
                             t_e_MeV=float(en[j]),cos_electron=float(cp[q]),t_positron_MeV=None))
        keep[m1]=False
    # --- pair
    m2=pm==2
    if m2.any():
        Ei=en[m2]; Tm=Ei-2*me
        Tminus=Tm*rng.random(Ei.size); Tplus=Tm-Tminus
        cpair=1-0.1*rng.random(Ei.size)   # electron forward-peaked (approximate)
        mi=np.where(vol&m2)[0]
        for j in mi:
            q=np.searchsorted(np.where(m2)[0],j)
            rows.append(dict(process="pair",e_in_MeV=float(en[j]),e_out_MeV=None,cos_scatter=None,
                             t_e_MeV=float(Tminus[q]),cos_electron=float(cpair[q]),t_positron_MeV=float(Tplus[q])))
        keep[m2]=False
        n2=m2.sum(); dA=iso_dir(n2)   # two 511 keV annihilation photons (positron annihilated at pair point)
        ann_pos.append(np.concatenate([pos[m2],pos[m2]])); ann_d.append(np.concatenate([dA,-dA])); ann_E.append(np.full(2*n2,me))
    keep &= newE>1e-3
    pos_new=[pos[keep]]; d_new=[newd[keep]]; E_new=[newE[keep]]
    for a,b,c in zip(ann_pos,ann_d,ann_E):
        pos_new.append(a); d_new.append(b); E_new.append(c)
    pos=np.concatenate(pos_new); dr=np.concatenate(d_new); en=np.concatenate(E_new)
    # Rayleigh/Compton events re-use position; no change
with open("events_%d.json"%seed,"w") as f: json.dump(rows,f)
print("iterations",it,"events in volume",len(rows))
'''
open('mc.py','w').write(code)
import subprocess,time
t=time.time()
r=subprocess.run(['python','mc.py','1','20000'],capture_output=True,text=True)
print(r.stdout[-2000:], r.stderr[-3000:], time.time()-t)
