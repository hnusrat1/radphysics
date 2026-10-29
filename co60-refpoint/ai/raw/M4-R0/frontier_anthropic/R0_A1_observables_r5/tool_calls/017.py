
code = r'''
import numpy as np, pickle, sys
sys.path.insert(0,'/tmp/work')
import mc, phys, ea
from phys import me, re
from sim import E_LINES
seg,hist,NTOT=pickle.load(open('/tmp/work/tallies.pkl','rb'))
V=np.pi*1.0**2*0.2
R=mc.Omega*105.0**2/(V*NTOT)     # tl*R = fluence per Phi0
E=seg['E']; tl=seg['tl']; prim=seg['prim'].astype(bool); ux,uy,uz=seg['ux'],seg['uy'],seg['uz']
w=tl*R
# --- cross sections at segment energies
muC=mc.mu_c(E); mupe=mc.mu_pe(E); mucoh=mc.mu_coh(E); mupair=mc.mu_pair(E)
mut=muC+mupe+mucoh+mupair
# energy-weighted transfer for Compton: tabulate
Eg=np.linspace(0.0005,1.45,3000)
mutrE_tab=np.array([phys.ne_w*__import__('scipy').integrate.quad(lambda c: phys.dsig_dOmega(e,c)*2*np.pi*e*(1-1/(1+e/me*(1-c))),-1,1)[0] for e in Eg])
mutrE=np.interp(E,Eg,mutrE_tab)
kerma_w = w*(mutrE + mupe*E + mupair*(E-2*me))   # MeV cm2/g * Phi0-normalized fluence
K_over_phi0 = 1.602176634e-10*kerma_w.sum()
out={}
# ---------- Layer 1
prim_tot=w[prim].sum(); scat_tot=w[~prim].sum(); tot=w.sum()
out['L1_primary_fluence_per_phi0']=prim_tot
out['L1_total_fluence_per_phi0']=tot
out['L1_scattered_fraction_fluence']=scat_tot/tot
out['L1_scattered_fraction_energy_fluence']=(w*E)[~prim].sum()/(w*E).sum()
out['L1_mean_photon_energy_MeV']=(w*E).sum()/tot
pe=np.arange(0,1.4+1e-9,0.05)
def binspec(mask):
    h,_=np.histogram(E[mask],bins=pe,weights=w[mask]); return h
out['L1_photon_fluence_spectrum_primary']=binspec(prim).tolist()
out['L1_photon_fluence_spectrum_scattered']=binspec(~prim).tolist()
cb=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
def angbin(mask):
    h,_=np.histogram(uz[mask],bins=cb,weights=w[mask]); return h/h.sum()
out['L1_angular_primary']=angbin(prim).tolist()
out['L1_angular_scattered']=angbin(~prim).tolist()
# ---------- Layer 2 (track-length estimators)
def fr(mask,comp):
    return comp[mask].sum()
def fracs(mask):
    tlm=tl[mask]
    d={'compton':(muC*tl)[mask].sum(),'photoelectric':(mupe*tl)[mask].sum(),
       'pair':(mupair*tl)[mask].sum(),'rayleigh':(mucoh*tl)[mask].sum()}
    s=sum(d.values()); return {k:v/s for k,v in d.items()}
def etfracs(mask):
    d={'compton':(mutrE*tl)[mask].sum(),'photoelectric':(mupe*E*tl)[mask].sum(),
       'pair':(mupair*(E-2*me)*tl)[mask].sum(),'rayleigh':0.0}
    s=sum(d.values()); return {k:v/s for k,v in d.items()}
allm=np.ones(E.size,bool)
out['L2_interaction_fractions']=fracs(allm)
out['L2_energy_transfer_fractions']=etfracs(allm)
out['L2_interaction_fractions_primary']=fracs(prim)
out['L2_interaction_fractions_scattered']=fracs(~prim)
out['L2_scattered_share_of_interactions']=(mut*tl)[~prim].sum()/(mut*tl).sum()
pickle.dump(dict(out=out,K_over_phi0=K_over_phi0,kerma_w=kerma_w),open('/tmp/work/part1.pkl','wb'))
print(K_over_phi0)
for k,v in out.items():
    print(k, v if not isinstance(v,list) else np.round(v,5))
'''
open('/tmp/work/an1.py','w').write(code)
import subprocess
r=subprocess.run(['python3','/tmp/work/an1.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-6000:], r.stderr[-3000:])
