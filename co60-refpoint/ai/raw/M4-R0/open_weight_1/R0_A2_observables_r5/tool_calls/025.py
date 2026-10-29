import numpy as np, pickle
with open('/tmp/photon_results.pkl','rb') as f:
    res=pickle.load(f)
N=6000000; V=np.pi*0.2
Omega=4*np.arcsin(np.sin(np.arctan(5/100))**2); Phi0=1/(Omega*105**2)
fl_prim=res['fl_prim']; fl_scat=res['fl_scat']
scale = 0.72948169572262/(fl_prim.sum())  # scale so primary per phi0 = analytic
spec_prim = fl_prim*scale     # fluence/phi0 per bin
spec_scat = fl_scat*scale
print('spec_prim sum', spec_prim.sum(), 'spec_scat sum', spec_scat.sum())
print('primary spectrum:', np.round(spec_prim,6))
print('scattered spectrum:', np.round(spec_scat,6))
# angular
angp = res['ang_prim']/res['ang_prim'].sum()
angs = res['ang_scat']/res['ang_scat'].sum()
print('angular primary:', np.round(angp,6))
print('angular scattered:', np.round(angs,6))
# Layer2
int_all=res['int_all']; int_prim=res['int_prim']; int_scat=res['int_scat']
etr_all=res['etr_all']; 
L2_if = {k:v for k,v in zip(['rayleigh','compton','photoelectric','pair'], int_all/int_all.sum())}
L2_etf = {k:v for k,v in zip(['rayleigh','compton','photoelectric','pair'], etr_all/etr_all.sum())}
print('L2_interaction_fractions', L2_if)
print('L2_energy_transfer_fractions', L2_etf)
print('L2_primary', int_prim/int_prim.sum())
print('L2_scattered', int_scat/int_scat.sum())
print('L2_scattered_share', int_scat.sum()/int_all.sum())
# Layer3
e_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.4])
Te=res['L3_Te']; cos_e=res['L3_cos']; par=res['L3_parent']
h=np.histogram(Te, bins=e_edges)[0].astype(float); h/=h.sum()
hp=np.histogram(Te[par==1], bins=e_edges)[0].astype(float); hp/=hp.sum()
print('L3 spectrum all:', np.round(h,5))
print('L3 spectrum prim:', np.round(hp,5))
print('L3 mean Te', Te.mean(), 'mean Te prim', Te[par==1].mean())
cbins=np.linspace(-1,1,11)
hd=np.histogram(cos_e, bins=cbins)[0].astype(float); hd/=hd.sum()
print('L3 initial direction:', np.round(hd,5))