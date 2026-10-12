
import pickle, numpy as np, sys
sys.path.insert(0,'.')
import mc2
allres=pickle.load(open('allres.pkl','rb'))
depths=[0.25,0.5,0.75,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0]
Ntot=12000000; V=mc2.Vcyl; P0=mc2.Phi0_hist; c5=depths.index(5.0)
tal={k:sum(r['tal'][k] for r in allres) for k in allres[0]['tal']}
Hfine=sum(r['Hfine'] for r in allres); Hcoarse=sum(r['Hcoarse'] for r in allres)
Hcos=sum(r['Hcos'] for r in allres); Esum=sum(r['Esum'] for r in allres); Wsum=sum(r['Wsum'] for r in allres)
fine=allres[0]['fine']; eedges=allres[0]['eedges']
re=2.8179403e-13; mc2_=0.51099895; NA=6.02214076e23
ne_w=0.55509*NA; ne_a=0.49919*NA
Kw=2*np.pi*re**2*mc2_*ne_w; Ka=2*np.pi*re**2*mc2_*ne_a
def Scol(T, I_eV, ne_per_g):
    tau=np.asarray(T)/mc2_; g=tau+1; b2=1-1/g**2
    Kg=2*np.pi*re**2*mc2_*ne_per_g
    Im=I_eV*1e-6/mc2_
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/g**2
    return Kg/b2*(np.log(tau**2*(tau+2)/(2*Im**2))+F)
def Yrad(T): return np.minimum(0.0069*T,0.5)
Ec=np.sqrt(fine[:-1]*fine[1:]); dE=np.diff(fine)
Scw=Scol(Ec,75.0,ne_w); Stw=Scw/(1-Yrad(Ec)); Srw=Stw-Scw
Scol_a=Scol(Ec,85.7,ne_a)
Qph=(Hfine[c5,0]+Hfine[c5,1])/(Ntot*V)/dE
Qph_p=(Hfine[c5,0])/(Ntot*V)/dE
b2c=1-1/(Ec/mc2_+1)**2
nb=len(Ec)
def slow(Q,S):
    qd=Q*dE
    cum=np.cumsum(qd[::-1])[::-1]
    return (cum-0.5*qd)/S
def delta_src(Phi):
    qd=np.zeros(nb); w=Phi*dE/b2c
    for j in range(nb):
        m=Ec>=2*Ec[j]
        qd[j]=Kw*np.sum(w[m])/Ec[j]**2
    return qd
Phi_ph=slow(Qph,Stw); Phi=Phi_ph.copy()
for it in range(40):
    Phi_new=slow(Qph+delta_src(Phi),Stw)
    d=np.max(np.abs(Phi_new-Phi))/np.max(Phi); Phi=Phi_new
    if d<1e-7: break
Phi_d=Phi-Phi_ph
m10=Ec>=0.01
Fel=np.sum((Phi*dE)[m10])/P0
print("electron fluence>10keV per Phi0", Fel)
delta_frac=np.sum((Phi_d*dE)[m10])/np.sum((Phi*dE)[m10])
print("delta fraction", delta_frac)
Emean=np.sum((Phi*dE*Ec)[m10])/np.sum((Phi*dE)[m10])
print("mean electron energy>10keV", Emean)
rc=np.sum((Phi*dE*Srw/(1-0)) [m10]) # placeholder
rc=np.sum((Phi*dE*(Stw-Scw))[m10])/np.sum((Phi*dE*Scw)[m10])
print("rad/col >10keV", rc)
g=np.sum(Phi*dE*(Stw-Scw))/np.sum(Phi*dE*Stw)
print("g (all electrons)", g)
# electron spectrum on coarse bins
cum=np.concatenate([[0],np.cumsum(Phi*dE)])
Fint=np.interp(eedges, fine, np.concatenate([[0],cum[1:]]) ) if False else None
# integrate Phi over coarse bins using log-interp of cumulative on fine edges
cumE=np.concatenate([[0],np.cumsum(Phi*dE)])  # at fine edges
def cum_at(x): return np.interp(x, fine, cumE)
coarse_fl=np.array([cum_at(eedges[i+1])-cum_at(eedges[i]) for i in range(len(eedges)-1)])/P0
print("coarse electron fluence bins sum", coarse_fl.sum(), Fel)
