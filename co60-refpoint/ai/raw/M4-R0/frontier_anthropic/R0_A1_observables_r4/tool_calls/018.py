
import sys; sys.path.insert(0,'/tmp/work')
import pickle, numpy as np, mc as MC, elec, phys
Tc,kerma,flu,ev,Ntot=pickle.load(open('/tmp/work/comb.pkl','rb'))
VOL=MC.VOL; F0=MC.F0
# fine log grid for electrons 10 keV .. 1.4 MeV, plus lower end for source
edges=np.exp(np.linspace(np.log(0.01),np.log(1.4),301))
cent=np.sqrt(edges[1:]*edges[:-1]); dE=np.diff(edges)
T=ev[:,0]; w=ev[:,2]
h,_=np.histogram(T,bins=edges,weights=w)
Qsrc=h/(Ntot*VOL*dE)       # per source photon, per cm3, per MeV
Ne=elec.WATER['ne']
Sres=elec.Sres_mass(cent,elec.WATER)*1.0   # MeV/cm (rho=1)
n=len(cent)
Phi=np.zeros(n); Qd=np.zeros(n)
# top-down
for i in range(n-1,-1,-1):
    # delta source at energy cent[i]: from parents E_k >= 2*cent[i]
    mask=cent>=2*cent[i]
    if i<n-1:
        ks=np.where(cent>=2*cent[i])[0]
        if ks.size>0:
            Qd[i]=np.sum(Phi[ks]*Ne*elec.moller_dsig(cent[ks],np.full(ks.size,cent[i]))*dE[ks])
    # Phi_i = (1/S) * sum_{j>=i} (Qsrc_j + Qd_j) dE_j
    Phi[i]=np.sum((Qsrc[i:]+Qd[i:])*dE[i:])/Sres[i]
# separate delta part: Phi_delta = (1/S) sum_{j>=i} Qd_j dE_j
Phid=np.array([np.sum(Qd[i:]*dE[i:])/Sres[i] for i in range(n)])
Phi_s=Phi-Phid
F=F0
tot=np.sum(Phi*dE)*F; print('electron fluence >10keV per phi0',tot,' delta part', np.sum(Phid*dE)*F, 'frac', np.sum(Phid*dE)/np.sum(Phi*dE))
print('mean E (fluence weighted)', np.sum(Phi*cent*dE)/np.sum(Phi*dE))
print('Phi(E) samples per phi0:', [(round(cent[i],3), float('%.3g'%(Phi[i]*F))) for i in range(0,n,30)])
