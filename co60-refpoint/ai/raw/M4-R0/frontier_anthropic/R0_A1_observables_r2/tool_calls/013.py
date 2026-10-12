
import pickle, numpy as np, sys
sys.path.insert(0,'.')
import mc2
allres=pickle.load(open('allres.pkl','rb'))
depths=[0.25,0.5,0.75,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0]
Ntot=12000000
V=mc2.Vcyl; P0=mc2.Phi0_hist
c5=depths.index(5.0)
Hfine=sum(r['Hfine'] for r in allres); Hcoarse=sum(r['Hcoarse'] for r in allres)
Hcos=sum(r['Hcos'] for r in allres); Esum=sum(r['Esum'] for r in allres); Wsum=sum(r['Wsum'] for r in allres)
fine=allres[0]['fine']; eedges=allres[0]['eedges']
re=2.8179403e-13; mc2_=0.51099895; NA=6.02214076e23
ne_w=0.55509*NA; ne_a=0.49919*NA
Kw=2*np.pi*re**2*mc2_*ne_w   # MeV cm2/g  (=0.0852 per cm for rho=1)
def Scol(T, I_eV, ne_per_g):
    tau=np.asarray(T)/mc2_; g=tau+1; b2=1-1/g**2
    Kg=2*np.pi*re**2*mc2_*ne_per_g
    Im=I_eV*1e-6/mc2_
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/g**2
    return Kg/b2*(np.log(tau**2*(tau+2)/(2*Im**2))+F)
def Yrad(T): return np.minimum(0.0069*T,0.5)
def Stot(T): 
    y=Yrad(T); return Scol(T,75.0,ne_w)/(1-y)
Ec=np.sqrt(fine[:-1]*fine[1:]); dE=np.diff(fine)
b2c=1-1/(Ec/mc2_+1)**2
Scw=Scol(Ec,75.0,ne_w); Stw=Stot(Ec)
Qph=(Hfine[c5,0]+Hfine[c5,1])/(Ntot*V)/dE
nb=len(Ec)
def slow(Q):
    # Phi_i = (1/S_i) * sum_{j>i} Q_j dE_j + 0.5 Q_i dE_i  (fluence per MeV, per history)
    qd=Q*dE
    cum=np.cumsum(qd[::-1])[::-1]   # sum_{j>=i}
    Phi=(cum-0.5*qd)/Stw
    return Phi
def delta_src(Phi):
    # Q_d(T_j)= sum_i Phi_i dE_i K/(beta_i^2 T_j^2) for E_i >= 2 T_j
    qd=np.zeros(nb)
    w=Phi*dE/b2c   # per fine bin
    for j in range(nb):
        T=Ec[j]
        m=Ec>=2*T
        qd[j]=Kw*np.sum(w[m])/T**2
    return qd
Phi_ph=slow(Qph)
Phi=Phi_ph.copy()
for it in range(30):
    Qd=delta_src(Phi)
    Phi_new=slow(Qph+Qd)
    diff=np.max(np.abs(Phi_new-Phi))/np.max(Phi)
    Phi=Phi_new
    if diff<1e-6: break
print("iters",it,diff)
Phi_d=Phi-Phi_ph
m10=Ec>=0.01
Ntot_e=np.sum((Phi*dE)[m10])/P0
print("electron fluence >10 keV per Phi0:",Ntot_e, " delta part:", np.sum((Phi_d*dE)[m10])/np.sum((Phi*dE)[m10]))
print("mean electron energy >10keV:", np.sum((Phi*dE*Ec)[m10])/np.sum((Phi*dE)[m10]))
print("rad/col:", np.sum((Phi*dE*Stw-Phi*dE*Scw)[m10])/np.sum((Phi*dE*Scw)[m10]))
Kcol_slow=np.sum(Phi*dE*Scw)/P0*1.602e-10
Kslow=np.sum(Phi*dE*Stw)/P0*1.602e-10
print("Kcol_slow (Gy cm2/Phi0)", Kcol_slow, " K_slow", Kslow, " g=",1-np.sum(Phi*dE*Scw)/np.sum(Phi*dE*Stw))
