import numpy as np, pickle

with open('mc_results.pkl','rb') as f:
    d=pickle.load(f)
fine5=d['fine5']; score28_at5=d['score28_at5']

def load_xcom():
    with open('/tmp/data/xcom_water.txt') as f:
        lines=[l for l in f if not l.startswith('#') and l.strip()]
    x=np.loadtxt(lines); return x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6]
E_x,mu_coh,mu_incoh,mu_pe,mu_pairn,mu_paire,mu_tot=load_xcom()
def mu_interp(E,col): return np.exp(np.interp(np.log(E),np.log(E_x),np.log(col)))
def mup(E): return dict(compton=mu_interp(E,mu_incoh), pe=mu_interp(E,mu_pe), pair=mu_interp(E,mu_pairn+mu_paire), rayleigh=mu_interp(E,mu_coh), total=mu_interp(E,mu_tot))
def load_estar():
    with open('/tmp/data/estar_water.txt') as f:
        lines=[l for l in f if not l.startswith('#') and l.strip()]
    est=np.loadtxt(lines); return est[:,0],est[:,1],est[:,2],est[:,3],est[:,4],est[:,5],est[:,6]
Ee,Scol,Srad,Stot,csda,yld,den=load_estar()
def interp_e(E,col): return np.exp(np.interp(np.log(E),np.log(Ee),np.log(col)))
def scol_w(E): return interp_e(E,Scol)
def srad_w(E): return interp_e(E,Srad)

mc2=0.51099895; re2=7.9408e-26
def kn_dsig(E0,cosT):
    a=E0/mc2; Ep=E0/(1.0+a*(1.0-cosT)); r=Ep/E0
    return 0.5*re2*r**2*(r+1.0/r-(1.0-cosT**2))
def compton_Ep(E0,cosT):
    a=E0/mc2; return E0/(1.0+a*(1.0-cosT))

fine_edges=np.arange(0.01,1.405,0.005); nf=len(fine_edges)-1
fine_mid=(fine_edges[:-1]+fine_edges[1:])/2
sc_fine=fine5[:,:,1].sum(axis=1)
w1=score28_at5[23,0,:].sum(); w2=score28_at5[26,0,:].sum(); wlines=np.array([w1,w2]); Elines=np.array([1.1732,1.3325])

Tg=np.concatenate([np.linspace(0.01,0.1,100,endpoint=False), np.linspace(0.1,1.4,600)])
q=np.zeros(Tg.size)
def add_photon_fine(E,w,qarr):
    d=mup(E)
    th=np.linspace(1e-5,np.pi,20000); c=np.cos(th); st=np.sin(th)
    T=E-compton_Ep(E,c)
    dsdo=kn_dsig(E,c)*2*np.pi*st
    dTdth=np.gradient(T,th)
    pT=dsdo/np.abs(dTdth); sig=np.trapezoid(dsdo,th); pT=pT/sig
    order=np.argsort(T); Ts=T[order]; Ps=pT[order]
    pvals=np.interp(Tg,Ts,Ps,left=0,right=0)
    qarr += w*d['compton']*pvals
    idx=np.argmin(np.abs(Tg-E)); qarr[idx]+=w*d['pe']/(Tg[1]-Tg[0])
    Tp=(E-1.022)/2
    if Tp>0.01:
        idx=np.argmin(np.abs(Tg-Tp)); qarr[idx]+=2*w*d['pair']/(Tg[1]-Tg[0])
for E,w in zip(Elines,wlines): add_photon_fine(E,w,q)
for E,w in zip(fine_mid,sc_fine):
    if w>0: add_photon_fine(E,w,q)
q_m=q

S=scol_w(Tg); Srad=srad_w(Tg)
Ne_g=3.343e23
Delta=0.010  # MeV cutoff
def moller(Tp_,T):
    gamma=1+Tp_/mc2; beta2=1-1/gamma**2
    A=(2*np.pi*re2*mc2)/(beta2*Tp_)
    return np.maximum(A*(1.0/T**2 + 1.0/(Tp_-T)**2 - (2*gamma-1)/gamma**2/(T*(Tp_-T))),0.0)

# restricted stopping power: S - energy to hard secondaries (>Delta)
L_rest=S.copy()
for i,Tp_ in enumerate(Tg):
    if Tp_ > 2*Delta:
        Tarr=np.linspace(Delta, Tp_/2, 2000)
        hard=np.trapezoid(Tarr*Ne_g*moller(Tp_,Tarr),Tarr)
        L_rest[i]=max(S[i]-hard, 0.05)

Q=np.array([np.trapezoid(q_m[Tg>=t],Tg[Tg>=t]) if (Tg>=t).any() else 0.0 for t in Tg])
# solve SF: Phi(T) = (Q(T) + int_{2T}^{Tmax} Phi(T') Ne_g moller(T'->T) dT') / L_rest(T)
Phi=np.zeros(Tg.size)
nT=Tg.size
for i in range(nT-1,-1,-1):
    Ti=Tg[i]
    j0=np.searchsorted(Tg,2*Ti,side='left')
    if j0<nT:
        Tj=Tg[j0:]
        sec=Phi[j0:]*Ne_g*moller(Tj,Ti)
        src=np.trapezoid(sec,Tj)
    else:
        src=0.0
    Phi[i]=(Q[i]+src)/L_rest[i]

# no-delta comparison: Phi_nd = Q/S
Phi_nd=Q/S

el_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
def binf(Ph):
    out=np.zeros(len(el_edges)-1)
    for i in range(len(el_edges)-1):
        lo,hi=el_edges[i],el_edges[i+1]; sel=(Tg>=lo)&(Tg<hi)
        out[i]=np.trapezoid(Ph[sel],Tg[sel]) if sel.sum()>1 else 0
    return out
st_tot=binf(Phi); st_nd=binf(Phi_nd)
Ft=st_tot.sum(); Fn=st_nd.sum()
print("no-delta fluence:",Fn," SF-with-delta fluence:",Ft)
print("delta fraction:",(Ft-Fn)/Ft)
meanE=np.sum(st_tot*(el_edges[:-1]+el_edges[1:])/2)/Ft
print("mean electron energy:",meanE)
# energy check: dose = int Phi L_rest dT + track-end? approx int Phi*S would be
print("int Phi*S:", np.trapezoid(Phi*S,Tg), "Kcol:", np.trapezoid(q_m*Tg,Tg))
print("int Phi*L_rest:", np.trapezoid(Phi*L_rest,Tg))
rad_col=np.trapezoid(Phi*Srad,Tg)/np.trapezoid(Phi*S,Tg)
print("rad/coll:",rad_col)
print("spectrum:",np.round(st_tot,5))
np.save('L4_Phi_tot.npy',Phi); np.save('L4_spec.npy',st_tot); np.save('L4_Tg.npy',Tg); np.save('L4_Phi_nd.npy',Phi_nd)
print("min L_rest, S:", L_rest.min(), S.min())