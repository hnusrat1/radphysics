
import sys; sys.path.insert(0,'/tmp/work')
import pickle, numpy as np, mc as MC, elec
Tc,kerma,flu,ev,Ntot=pickle.load(open('/tmp/work/comb.pkl','rb'))
VOL=MC.VOL; F0=MC.F0; conv=1.602176634e-10  # MeV/g -> Gy
K_per_phi0 = kerma[0]/Ntot*F0/VOL*conv   # Gy cm2 per Phi0  (kerma = energy transfer incl pe,pair)
print('K/Phi0 [Gy cm2] =',K_per_phi0)
# electron-based collision energy: sum Phi_e * Scol  (electrons >= 10 keV) + events below 10 keV
edges=np.exp(np.linspace(np.log(0.01),np.log(1.4),301)); cent=np.sqrt(edges[1:]*edges[:-1]); dE=np.diff(edges)
# (re-run recursion quickly)
T=ev[:,0]; w=ev[:,2]
h,_=np.histogram(T,bins=edges,weights=w); Qsrc=h/(Ntot*VOL*dE)
Ne=elec.WATER['ne']; Sres=elec.Sres_mass(cent,elec.WATER); Scol=elec.Scol_per_mass(cent,elec.WATER); Srad=elec.Srad_per_mass(cent)
n=len(cent); Phi=np.zeros(n); Qd=np.zeros(n)
for i in range(n-1,-1,-1):
    if i<n-1:
        ks=np.where(cent>=2*cent[i])[0]
        if ks.size>0: Qd[i]=np.sum(Phi[ks]*Ne*elec.moller_dsig(cent[ks],np.full(ks.size,cent[i]))*dE[ks])
    Phi[i]=np.sum((Qsrc[i:]+Qd[i:])*dE[i:])/Sres[i]
Phid=np.array([np.sum(Qd[i:]*dE[i:])/Sres[i] for i in range(n)])
Fl=Phi*F0; Fld=Phid*F0   # per Phi0
Ecol=np.sum(Fl*Scol*dE)*conv
Erad=np.sum(Fl*Srad*dE)*conv
Elow=np.sum(ev[(T<0.01)*1>0 if False else (T<0.01),0]*ev[(T<0.01),2])*F0/(Ntot*VOL)*conv
print('electron-based Kcol(>10keV)',Ecol,' low-energy events',Elow,' sum',Ecol+Elow)
print('ratio electron-based/photon-level collision-ish', (Ecol+Elow)/K_per_phi0)
print('Erad',Erad,'g approx', Erad/(Erad+Ecol))
# energy in source events above 10 keV (should be source-energy budget)
Ein=np.sum(T*w)*F0/(Ntot*VOL)*conv
print('total initial electron energy per Phi0 (all events, Compton+pe+pair):',Ein)
