
l4 = r'''
import sys; sys.path.insert(0,'/tmp/work')
import pickle, numpy as np, mc as MC, elec
Tc,kerma,flu,ev,Ntot=pickle.load(open('/tmp/work/comb.pkl','rb'))
VOL=MC.VOL; F0=MC.F0; conv=1.602176634e-10
K_per_phi0 = kerma[0]/Ntot*F0/VOL*conv
edges=np.exp(np.linspace(np.log(0.01),np.log(1.4),301)); cent=np.sqrt(edges[1:]*edges[:-1]); dE=np.diff(edges)
T=ev[:,0]; w=ev[:,2]
h,_=np.histogram(T,bins=edges,weights=w); Qsrc=h/(Ntot*VOL*dE)
Ne=elec.WATER['ne']
Sres=elec.Sres_mass(cent,elec.WATER); Scol=elec.Scol_per_mass(cent,elec.WATER); Srad=elec.Srad_per_mass(cent)
n=len(cent); Phi=np.zeros(n); Qd=np.zeros(n)
for i in range(n-1,-1,-1):
    if i<n-1:
        ks=np.where(cent>=2*cent[i])[0]
        if ks.size>0: Qd[i]=np.sum(Phi[ks]*Ne*elec.moller_dsig(cent[ks],np.full(ks.size,cent[i]))*dE[ks])
    Phi[i]=np.sum((Qsrc[i:]+Qd[i:])*dE[i:])/Sres[i]
Phid=np.array([np.sum(Qd[i:]*dE[i:])/Sres[i] for i in range(n)])
Fl=Phi*F0; Fld=Phid*F0
Ered=np.sum(Fl*Sres*dE)*conv
Ecol=np.sum(Fl*Scol*dE)*conv
Ein_src=np.sum(Qsrc*cent*dE)*VOL*F0*conv  # not used
Ein=np.sum(T*w)*F0/(Ntot*VOL)*conv
Elow=np.sum(T[T<0.01]*w[T<0.01])*F0/(Ntot*VOL)*conv
print('K_per_phi0',K_per_phi0,'Ein(all source events)',Ein)
print('sum Phi*Sres',Ered,'ratio to photon K',Ered/K_per_phi0)
print('sum Phi*Scol',Ecol,'ratio',Ecol/K_per_phi0)
print('delta frac',np.sum(Phid*dE)/np.sum(Phi*dE),' Ldelta part of Scol:', np.sum(Fl*elec.Ldelta_per_e(cent)*Ne*dE)*conv)
'''
open('/tmp/work/l4.py','w').write(l4)
import subprocess
print(subprocess.run(['python','/tmp/work/l4.py'],capture_output=True,text=True,cwd='/tmp/work'))
