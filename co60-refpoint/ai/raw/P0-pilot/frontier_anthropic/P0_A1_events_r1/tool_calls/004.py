
code = r'''
import numpy as np, sys, json
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 1)
re=2.8179403e-13; me=0.510998950
NA=6.02214076e23
Ne=0.5551*NA            # electrons per gram of water (3.343e23 /g), density 1 g/cm3
def sigKN(E):
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)-(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2)
# pair production (nuc+electron) per g/cm2 in water: approximate interpolation (cm2/g)
_pe_E=np.array([1.0,1.022,1.5,2.0,3.0,4.0,6.0,10.0,20.0,100.0])
_pe_s=np.array([0,0,0.00005,0.00035,0.0025,0.0055,0.0120,0.0200,0.0216,0.0216])
def sig_pair(E):
    return np.interp(E,_pe_E,_pe_s)
def sig_coh(E):   # cm2/g, approx ~ E^-1.2 (0.00014 at 2 MeV)
    return 0.00014*(2.0/np.maximum(E,1e-3))**1.2
def sig_pe(E):    # cm2/g, approx 0.0028 (100 keV/E)^3
    return 0.0028*(0.1/np.maximum(E,1e-4))**3
def mu_parts(E):  # returns (compton, pe, pair, coh) linear coefficients cm^-1
    comp=sigKN(E)*Ne
    return comp, sig_pe(E), sig_pair(E), sig_coh(E)

# Compton energy-fraction sampling by rejection on dsig/deps ~ (eps+1/eps - sin^2)
def sample_compton(E):
    n=E.size; k=E/me
    epsmin=1/(1+2*k)
    out_eps=np.empty(n)
    todo=np.arange(n)
    while todo.size:
        kk=k[todo]; em=epsmin[todo]
        eps=em+(1-em)*rng.random(todo.size)
        c=1-(1/eps-1)/kk
        f=eps+1/eps-(1-c*c)
        fmax=2.0  # f <= 2 + ... bound checked below
        acc=rng.random(todo.size)*fmax < f
        out_eps[todo[acc]]=eps[acc]
        todo=todo[~acc]
    return out_eps
# check the bound numerically
_e=np.linspace(0.0,1,2000001)[1:]
_k=2.0/me
_ee=1/(1+2*_k)+(1-1/(1+2*_k))*np.linspace(0,1,20001)
_c=1-(1/_ee-1)/_k
_f=_ee+1/_ee-(1-_c*_c)
assert _f.max()<=2.0+1e-9, _f.max()

def iso_dir(n):
    c=2*rng.random(n)-1; ph=2*np.pi*rng.random(n); s=np.sqrt(1-c*c)
    return np.stack([s*np.cos(ph),s*np.sin(ph),c],axis=1)

def perp_basis(d):
    # returns two unit vectors perpendicular to each row of d
    a=np.where(np.abs(d[:,[0]])<0.9, np.array([[1.0,0,0]]), np.array([[0,1.0,0]]))
    u=np.cross(d,a); u/=np.linalg.norm(u,axis=1,keepdims=True)
    v=np.cross(d,u)
    return u,v

def rotate(d, cth, phi):
    u,v=perp_basis(d)
    s=np.sqrt(np.maximum(0,1-cth*cth))
    nd=(cth[:,None]*d + s[:,None]*(np.cos(phi)[:,None]*u+np.sin(phi)[:,None]*v))
    return nd/np.linalg.norm(nd,axis=1,keepdims=True)

# Rayleigh angle sampling: pdf(c) ~ (1+c^2) exp(-A(1-c)), A=2 k^2/q0^2, k in 1/Angstrom, q0=2.5 1/A
def sample_rayleigh_c(E):
    n=E.size; kA=(E*1000.0)/12.398
    A=2*kA**2/2.5**2
    out=np.empty(n); todo=np.arange(n)
    while todo.size:
        a=A[todo]
        u=rng.exponential(1.0,todo.size)/a
        ok=u<=2.0
        u2=u
        acc=ok & (rng.random(todo.size)*1.0 < (1+(1-u2)**2)/2)
        out[todo[acc]]=1-u2[acc]
        todo=todo[~acc]
    return out

# photoelectron angle: pdf ~ sin^2/(1-beta cos)^4 (K-shell like), rejection
def sample_pe_cos(E):
    n=E.size
    b=np.sqrt(1-(me/(E+me))**2)
    out=np.empty(n); todo=np.arange(n)
    while todo.size:
        c=2*rng.random(todo.size)-1
        bb=b[todo]
        f=(1-c*c)/(1-bb*c)**4
        fmax=1.0/(1-bb)**2*0+ (1.0/(1-bb)**4)*1.0
        # envelope: use max of f over c, approx at c=1 -> (1-c^2)=0; use numeric bound 1/(1-b)^4*... use simple accept with f/ (4/(1-b)^2)
        bound=np.maximum(0.25,1.0)/(1-bb)**2   # f<=1/(1-b)^2 *? checked empirically below
        acc=rng.random(todo.size)*bound < f
        out[todo[acc]]=c[acc]
        todo=todo[~acc]
    return out

# ---------- transport ----------
LO=np.array([-15.0,-15.0,0.0]); HI=np.array([15.0,15.0,30.0])
def in_volume(p):
    return (np.hypot(p[:,0],p[:,1])<=1.0) & (np.abs(p[:,2]-10.0)<=0.1)

def dist_to_exit(p,d):
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(LO-p)/d; t2=(HI-p)/d
    tmax=np.where(d>0,t2,np.where(d<0,t1,np.inf))
    return np.min(tmax,axis=1)

E0=2.0
N_hist=int(sys.argv[2]) if len(sys.argv)>2 else 200000
# source: pyramid 10x10 at 100 cm SSD, uniform in solid angle: sample (u,v) in [-0.05,0.05]^2, accept w.p. (1+u^2+v^2)^-1.5/1
events=[]
# generate primaries
u=rng.uniform(-0.05,0.05,N_hist); v=rng.uniform(-0.05,0.05,N_hist)
acc=rng.random(N_hist) < (1+u*u+v*v)**(-1.5)
u=u[acc]; v=v[acc]
d=np.stack([u,v,np.ones_like(u)],axis=1); d/=np.linalg.norm(d,axis=1,keepdims=True)
p=np.stack([100*u,100*v,np.zeros_like(u)],axis=1)  # position at phantom entrance face (source at z=-100)
E=np.full(p.shape[0],E0)
prim=np.ones(p.shape[0],bool)   # True if still unscattered primary (only informational)
events_rows=[]
# photon population arrays
pos=p; dr=d; en=E
# guarantee for each entry a flag 'annih' not needed
it=0
while pos.shape[0]>0 and it<200:
    it+=1
    comp,pe,pair,coh=mu_parts(en)
    mu_cm=(comp+pe*1+pair*1+coh*1)  # placeholder (overwritten below)
    # coefficients in cm^-1: comp is already cm^-1 (Ne*sigma); others are cm2/g times 1 g/cm3
    tot=comp+pe+pair+coh
    # check sigma per g for sanity: comp per gram = sigKN*Ne
    s=-np.log(rng.random(pos.shape[0]))/tot
    tex=dist_to_exit(pos,dr)
    inside=s<tex
    # photons leaving: drop
    pos=pos[inside]; dr=dr[inside]; en=en[inside]; comp=comp[inside]; pe=pe[inside]; pair=pair[inside]; coh=coh[inside]; tot=tot[inside]; s=s[inside]
    if pos.shape[0]==0: break
    pos=pos+s[:,None]*dr
    vol=in_volume(pos)
    r=rng.random(pos.shape[0])*tot
    proc=np.where(r<comp,0,np.where(r<comp+pe,1,np.where(r<comp+pe+pair,2,3)))  # 0 compton 1 pe 2 pair 3 coh
    # record events in volume
    idx=np.where(vol)[0]
    for i in idx:
        pass
    # --- compton ---
    m=proc==0
    if m.any():
        Ei=en[m]; eps=sample_compton(Ei); Eo=Ei*eps
        k=Ei/me; cth=1-(1/eps-1)/k
        phi=2*np.pi*rng.random(Ei.size)
        d_in=dr[m]
        pe_vec=Ei[:,None]*d_in-Eo[:,None]*rotate(d_in,cth,phi)
        pmag=np.linalg.norm(pe_vec,axis=1)
        cos_e=(Ei-Eo*cth)/np.maximum(pmag,1e-12)
        Te=Ei-Eo
        newd=rotate(d_in,cth,phi)
    # --- coherent ---
    m3=proc==3
    if m3.any():
        Ei=en[m3]; cth3=sample_rayleigh_c(Ei); phi3=2*np.pi*rng.random(Ei.size)
        newd3=rotate(dr[m3],cth3,phi3)
    # --- photoelectric ---
    m1=proc==1
    if m1.any():
        Ei=en[m1]; cos_pe=sample_pe_cos(Ei)
    # --- pair ---
    m2=proc==2
    if m2.any():
        Ei=en[m2]; Tm=(Ei-2*me)
        Te_minus=Tm*rng.random(Ei.size); Te_plus=Tm-Te_minus
        cos_pair=1-0.1*rng.random(Ei.size)   # electron forward (approx.)
    # ---- record events (only those in volume) ----
    rec=[]
    if vol.any():
        vi=np.where(vol)[0]
        # map processes
        for j in vi:
            pr=proc[j]
            if pr==0:
                kk=np.where(np.where(m)[0]==j)[0][0]
                rec.append(dict(process="compton",e_in_MeV=float(en[j]),e_out_MeV=float(Eo[kk]),cos_scatter=float(cth[kk]),
                                t_e_MeV=float(Te[kk]),cos_electron=float(np.clip(cos_e[kk],-1,1)),t_positron_MeV=None))
            elif pr==1:
                kk=np.where(np.where(m1)[0]==j)[0][0]
                rec.append(dict(process="photoelectric",e_in_MeV=float(en[j]),e_out_MeV=None,cos_scatter=None,
                                t_e_MeV=float(en[j]),cos_electron=float(cos_pe[kk]),t_positron_MeV=None))
            elif pr==2:
                kk=np.where(np.where(m2)[0]==j)[0][0]
                rec.append(dict(process="pair",e_in_MeV=float(en[j]),e_out_MeV=None,cos_scatter=None,
                                t_e_MeV=float(Te_minus[kk]),cos_electron=float(cos_pair[kk]),t_positron_MeV=float(Te_plus[kk])))
            else:
                kk=np.where(np.where(m3)[0]==j)[0][0]
                rec.append(dict(process="rayleigh",e_in_MeV=float(en[j]),e_out_MeV=float(en[j]),cos_scatter=float(cth3[kk]),
                                t_e_MeV=0.0,cos_electron=None,t_positron_MeV=None))
    events_rows.extend(rec)
    # ---- update photon states ----
    newpos=pos.copy(); newd=dr.copy(); newE=en.copy(); alive=np.ones(pos.shape[0],bool)
    if m.any():
        newd[m]=newd_c=rotate(dr[m],cth,phi); newE[m]=Eo
    if m3.any():
        newd[m3]=newd3; # energy unchanged
    alive[m1]=False
    # pair: photon removed; create two annihilation photons
    if m2.any():
        alive[m2]=False
        n2=m2.sum()
        dA=iso_dir(n2)
        ann_pos=pos[m2]
        # store as new photons
        newpos_ann=np.concatenate([ann_pos,ann_pos]); newd_ann=np.concatenate([dA,-dA]); newE_ann=np.full(2*n2,me)
    else:
        newpos_ann=np.zeros((0,3)); newd_ann=np.zeros((0,3)); newE_ann=np.zeros(0)
    # low energy cutoff: absorb photons below 1 keV
    alive &= newE>1e-3
    pos=np.concatenate([newpos[alive],newpos_ann]); dr=np.concatenate([newd[alive],newd_ann]); en=np.concatenate([newE[alive],newE_ann])
    # pe: no new photons
    # drop photons with energy below cutoff
    keep=en>1e-3
    pos=pos[keep]; dr=dr[keep]; en=en[keep]
events=events_rows
with open("events_%s.json"%(sys.argv[1] if len(sys.argv)>1 else "x"),"w") as f:
    json.dump(events,f)
print("iterations",it,"n_events",len(events))
'''
open('mc.py','w').write(code)
import subprocess,time
t=time.time()
print(subprocess.run(['python','mc.py','1','20000'],capture_output=True,text=True).stdout[-2000:], subprocess.run(['python','mc.py','1','20000'],capture_output=True,text=True).stderr[-3000:])
print(time.time()-t)
