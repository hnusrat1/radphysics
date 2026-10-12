import numpy as np
Phi_nd=np.load('L4_Phi_nd.npy'); Phi_tot=np.load('L4_Phi_tot.npy'); Tg=np.load('L4_Tg.npy')
# recompute one-generation quickly with saved Phi_nd? Actually L4_Phi_tot is the diverged full one. Recompute one-gen.
# load everything needed
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
q=np.zeros(Tg.size)
def add_photon_fine(E,w,qarr):
    d=mup(E)
    th=np.linspace(1e-5,np.pi,20000); c=np.cos(th); st=np.sin(th)
    T=E-compton_Ep(E,c); dsdo=kn_dsig(E,c)*2*np.pi*st; dTdth=np.gradient(T,th)
    pT=dsdo/np.abs(dTdth); sig=np.trapezoid(dsdo,th); pT=pT/sig
    order=np.argsort(T); Ts=T[order]; Ps=pT[order]
    qarr += w*d['compton']*np.interp(Tg,Ts,Ps,left=0,right=0)
    idx=np.argmin(np.abs(Tg-E)); qarr[idx]+=w*d['pe']/(Tg[1]-Tg[0])
    Tp=(E-1.022)/2
    if Tp>0.01:
        idx=np.argmin(np.abs(Tg-Tp)); qarr[idx]+=2*w*d['pair']/(Tg[1]-Tg[0])
for E,w in zip(Elines,wlines): add_photon_fine(E,w,q)
for E,w in zip(fine_mid,sc_fine):
    if w>0: add_photon_fine(E,w,q)
S=scol_w(Tg); Ne_g=3.343e23
def moller(Tp_,T):
    gamma=1+Tp_/mc2; beta2=1-1/gamma**2
    A=(2*np.pi*re2*mc2)/(beta2*Tp_)
    return np.maximum(A*(1.0/T**2+1.0/(Tp_-T)**2-(2*gamma-1)/gamma**2/(T*(Tp_-T))),0.0)
Q=np.array([np.trapezoid(q[Tg>=t],Tg[Tg>=t]) if (Tg>=t).any() else 0.0 for t in Tg])
Phi_nd=Q/S
q_sec=np.zeros(Tg.size); nT=Tg.size
for i,Ti in enumerate(Tg):
    j0=np.searchsorted(Tg,2*Ti,side='left')
    if j0<nT:
        Tj=Tg[j0:]; q_sec[i]=np.trapezoid(Phi_nd[j0:]*Ne_g*moller(Tj,Ti),Tj)
Q_sec=np.array([np.trapezoid(q_sec[Tg>=t],Tg[Tg>=t]) if (Tg>=t).any() else 0.0 for t in Tg])
Phi_delta=Q_sec/S
Phi_onergen=Phi_nd+Phi_delta
el_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
def binf(Ph):
    out=np.zeros(len(el_edges)-1)
    for i in range(len(el_edges)-1):
        lo,hi=el_edges[i],el_edges[i+1]; sel=(Tg>=lo)&(Tg<hi)
        out[i]=np.trapezoid(Ph[sel],Tg[sel]) if sel.sum()>1 else 0
    return out
b_nd=binf(Phi_nd); b_dl=binf(Phi_delta); b_tot=binf(Phi_onergen)
print("bin   no-delta   delta    total")
for i in range(25):
    if b_tot[i]>1e-6:
        print(f"{el_edges[i]:.2f}-{el_edges[i+1]:.2f}: {b_nd[i]:.6f} {b_dl[i]:.6f} {b_tot[i]:.6f}")
print("totals:",b_nd.sum(),b_dl.sum(),b_tot.sum(),"delta fraction:",b_dl.sum()/b_tot.sum())
print("int(Phi_tot*S):", np.trapezoid(Phi_onergen*S,Tg),"Kcol:",np.trapezoid(q*Tg,Tg))
# save one-gen as final L4
np.save('L4_Phi_tot.npy',Phi_onergen); np.save('L4_spec.npy',b_tot); np.save('L4_Phi_nd.npy',Phi_nd); np.save('L4_delta_spec.npy',b_dl)