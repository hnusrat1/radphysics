
code_final2 = r'''
import sys, pickle
sys.path.insert(0,'/tmp/work')
import numpy as np
import phys, deposit, electrons, sources
from phys import mu_all, Yrad, Stot, Scol, Srad, EE, E_
S_phi0=0.009975072683099257*110.0**2; V=np.pi*1.0**2*0.2; Ntot=4000000
D=pickle.load(open('/tmp/work/photon_tallies.pkl','rb'))
tot=D['tot']; seg=D['seg']
L4=pickle.load(open('/tmp/work/L4.pkl','rb'))
EBe=electrons.EBe
CB=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
Phi_p=float(np.exp(-mu_all(np.array([2.0]))[4][0]*10))
rng=np.random.default_rng(5)
cat=sources.build_sources(seg,Ntot,rng,Phi_p)
T=cat['T']; w=cat['w']; uz=cat['u'][:,2]; ps=cat['ps']; par=cat['par']; q=cat['q']
ok_all=np.ones(len(T),bool)
# ---- Layer 3
def binspec(mask):
    tt=T[mask]; ww=w[mask]
    b=np.digitize(tt,EBe)-1
    good=(b>=0)&(b<25)&(tt>=0.01)
    s=np.zeros(25); np.add.at(s,b[good],ww[good])
    return s/s.sum()
L3_spec=binspec(np.ones(len(T),bool))
L3_spec_p=binspec(par==0)
L3_mean=float((w*T).sum()/w.sum())
L3_mean_p=float((w*T)[par==0].sum()/w[par==0].sum())
cb=np.clip(np.digitize(uz,CB)-1,0,9)
dirw=np.zeros(10); np.add.at(dirw,cb,w); L3_dir=dirw/dirw.sum()
# ---- kerma
Y=Yrad(np.maximum(T,0.01))
K_all=1.602e-10*(w*T).sum()
Kcol_all=1.602e-10*(w*T*(1-Y)).sum()
g=1-Kcol_all/K_all
beta=1.0
D_all=beta*Kcol_all
dosefrac_scat=float((w*T*(1-Y))[ps].sum()/(w*T*(1-Y)).sum())
# primary-only Kcol check
Kcol_prim=1.602e-10*(w*T*(1-Y))[par==0].sum()
print("K/Phi0",K_all,"Kcol/Phi0",Kcol_all,"g",g,"dose frac scat",dosefrac_scat,"prim Kcol",Kcol_prim)
dep2=deposit.transfer_coeffs(np.array([2.0]))
print("analytic prim Kcol", 1.602e-10*Phi_p*float(dep2['col'][0]))
# scattered Kcol check at z=10 from tot['Kd']
ZD=np.arange(0.25,20.0001,0.25)
Kd=tot['Kd']; k10=np.argmin(np.abs(ZD-10.0))
print("scattered Kcol at 10cm analytic-tally", 1.602e-10*Kd[k10], "sources", 1.602e-10*(w*T*(1-Y))[ps].sum())
# ---- L4 outputs
tb=L4['tb']; tcos=L4['tcos']; tot_track=L4['tot']
dw=np.diff(EBe)
psi=tb/dw  # per MeV per Phi0
L4=dict(
 L4_electron_fluence_per_phi0=float(tot_track),
 L4_electron_fluence_spectrum=list(tb),
 L4_delta_fraction=float(L4_dummy) if False else float(L4['delt']/L4['tot']) if isinstance(L4,dict) else 0.0,
)
'''
print("written")
