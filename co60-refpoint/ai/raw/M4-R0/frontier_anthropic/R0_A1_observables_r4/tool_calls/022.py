
l45 = r'''
import sys; sys.path.insert(0,'/tmp/work')
import pickle, numpy as np, mc as MC, elec
Tc,kerma,flu,ev,Ntot=pickle.load(open('/tmp/work/comb.pkl','rb'))
VOL=MC.VOL; F0=MC.F0; conv=1.602176634e-10
K_tot=kerma[0]/Ntot*F0/VOL*conv           # K/Phi0 (Gy cm2)
K_100=kerma[1]/Ntot*F0/VOL*conv
edges=np.exp(np.linspace(np.log(0.01),np.log(1.4),301)); cent=np.sqrt(edges[1:]*edges[:-1]); dE=np.diff(edges)
T=ev[:,0]; w=ev[:,2]
h,_=np.histogram(T,bins=edges,weights=w); Qsrc=h/(Ntot*VOL*dE)
Ne=elec.WATER['ne']
Sres=elec.Sres_mass(cent,elec.WATER); Scol=elec.Scol_per_mass(cent,elec.WATER); Srad=elec.Srad_per_mass(cent)
Sres_air=elec.Sres_mass(cent,elec.AIR)
n=len(cent); Phi=np.zeros(n); Qd=np.zeros(n)
for i in range(n-1,-1,-1):
    if i<n-1:
        ks=np.where(cent>=2*cent[i])[0]
        if ks.size>0: Qd[i]=np.sum(Phi[ks]*Ne*elec.moller_dsig(cent[ks],np.full(ks.size,cent[i]))*dE[ks])
    Phi[i]=np.sum((Qsrc[i:]+Qd[i:])*dE[i:])/Sres[i]
Phid=np.array([np.sum(Qd[i:]*dE[i:])/Sres[i] for i in range(n)])
Ered=np.sum(Phi*Sres*dE)*conv*F0
Elow=np.sum(T[T<0.01]*w[T<0.01])*F0/(Ntot*VOL)*conv
# energy-balance normalisation: sum Phi*Sres*(scale) + low-energy part = K_col
Kcol_target=K_tot*(1.0)   # collision part; radiative correction below
Erad_unscaled=np.sum(Phi*Srad*dE)*conv*F0
g_est=Erad_unscaled/(Erad_unscaled+Kcol_target)
Kcol=K_tot*(1-g_est)
scale=(Kcol-Elow)/(Ered)
Fl=Phi*F0*scale; Fld=Phid*F0*scale
Erad=Erad_unscaled*scale
g=Erad/(Erad+Kcol)
# L4 outputs
Lfl=np.sum(Fl*dE)
e_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
cum=np.concatenate([[0],np.cumsum(Fl*dE)])
ce_cum=np.interp(e_edges,np.concatenate([[edges[0]],edges[1:]]),cum)  # approx cumulative at fine edges
spec=np.diff(np.interp(e_edges,edges,np.concatenate([[0],np.cumsum(Fl*dE)])))
print('L4 total',Lfl,'sum spec',spec.sum())
print('spec',np.round(spec,6).tolist())
print('delta frac',np.sum(Fld*dE)/Lfl)
print('mean E',np.sum(Fl*cent*dE)/Lfl)
print('rad/col (fluence weighted)',np.sum(Fl*Srad*dE)/np.sum(Fl*Scol*dE))
print('g',g,'Kcol',Kcol,'K',K_tot,'Kcol check Ered+Elow',Ered*scale+Elow)
# Spencer-Attix
D=0.01
Sw=Sres; Sa=Sres_air
phi_D=Phi[0]*scale
sw=(np.sum(Phi*scale*Sw*dE)+D*phi_D*Sw[0])
sa=(np.sum(Phi*scale*Sa*dE)+D*phi_D*Sa[0])
print('s_w_air (track-end incl)',sw/sa, ' without track end',np.sum(Phi*Sw*dE)/np.sum(Phi*Sa*dE))
print('PDD5 %',100*K_tot/K_100, 'K105',K_tot,'K100.5',K_100)
pickle.dump(dict(spec=spec.tolist(),Lfl=Lfl,delta=np.sum(Fld*dE)/Lfl,meanE=np.sum(Fl*cent*dE)/Lfl,
  rc=np.sum(Fl*Srad*dE)/np.sum(Fl*Scol*dE),g=g,Kcol=Kcol,K=K_tot,sw=sw/sa,pdd5=100*K_tot/K_100,
  Erad=Erad,Elow=Elow), open('/tmp/work/L45.pkl','wb'))
'''
open('/tmp/work/l45.py','w').write(l45)
import subprocess
r=subprocess.run(['python','/tmp/work/l45.py'],capture_output=True,text=True,cwd='/tmp/work'); print(r.stdout, r.stderr[-2000:])
