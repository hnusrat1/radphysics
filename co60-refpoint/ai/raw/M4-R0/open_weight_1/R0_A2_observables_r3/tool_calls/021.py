import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import json

# ---------------- data ----------------
d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
_Em=d[:,0]; _coh=d[:,1]; _incoh=d[:,2]; _pe=d[:,3]; _pn=d[:,4]; _pef=d[:,5]; _tot=d[:,6]
f_coh=interp1d(_Em,_coh,bounds_error=False,fill_value=(_coh[0],_coh[-1]))
f_incoh=interp1d(_Em,_incoh,bounds_error=False,fill_value=(_incoh[0],_incoh[-1]))
f_pe=interp1d(_Em,_pe,bounds_error=False,fill_value=(_pe[0],_pe[-1]))
f_pair=interp1d(_Em,_pn+_pef,bounds_error=False,fill_value=(_pn[0]+_pef[0],_pn[-1]+_pef[-1]))
f_tot=interp1d(_Em,_tot,bounds_error=False,fill_value=(_tot[0],_tot[-1]))

d2=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
_Ee=d2[:,0]; _Sc=d2[:,1]; _Sr=d2[:,2]; _St=d2[:,3]; _R=d2[:,4]; _Y=d2[:,5]; _de=d2[:,6]
f_Sc=interp1d(_Ee,_Sc,bounds_error=False,fill_value=(_Sc[0],_Sc[-1]))
f_Sr=interp1d(_Ee,_Sr,bounds_error=False,fill_value=(_Sr[0],_Sr[-1]))
f_St=interp1d(_Ee,_St,bounds_error=False,fill_value=(_St[0],_St[-1]))
f_R=interp1d(_Ee,_R,bounds_error=False,fill_value=(_R[0],_R[-1]))
f_Y=interp1d(_Ee,_Y,bounds_error=False,fill_value=(_Y[0],_Y[-1]))

# ---------------- constants & grids ----------------
me=0.51099895
r0=2.8179403e-13; NA=6.02214076e23
Ne_w=NA*(10.0/18.015268)
C_GY=1.602e-10

E1=1.1732; E2=1.3325
Z=5.0
PH_EDGES=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
                   0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
EL_EDGES=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
                   0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
CT_EDGES=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
phc=(PH_EDGES[:-1]+PH_EDGES[1:])/2
elc=(EL_EDGES[:-1]+EL_EDGES[1:])/2

def kn_ratio(E,mu):
    a=E/me; r=1/(1+a*(1-mu)); return r*r*(r+1/r-(1-mu**2))

def mu_tr(E):
    # average compton electron energy via integration
    a=E/me
    num,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu)*(E - E/(1+a*(1-mu))),-1,1)
    den,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu),-1,1)
    fC=num/den/E if den>0 else 0
    return f_incoh(E)*fC+f_pe(E)+(f_pair(E)*(1-1.022/E) if E>1.022 else 0)
def mu_en(E):
    mt=mu_tr(E)
    a=E/me
    # average electron energies for g
    TeC,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu)*(E-E/(1+a*(1-mu))),-1,1)
    den,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu),-1,1)
    TeC=TeC/den if den>0 else 0
    gC=f_Y(max(TeC,0.011)); gPE=f_Y(E); gpair=f_Y(max((E-1.022)/2,0.011)) if E>1.022 else 0
    fC=TeC/E if E>0 else 0
    gbar=(f_incoh(E)*fC*gC+f_pe(E)*gPE+(f_pair(E)*(1-1.022/E)*gpair if E>1.022 else 0))/max(mt,1e-30)
    return mt*(1-gbar)

# ---------------- load MC ----------------
r=np.load('mc_results.npz',allow_pickle=True)
depths=r['depths']; prim_tl=r['prim_tl']; scat_tl=r['scat_tl']
spec_prim=r['spec_prim']; spec_scat=r['spec_scat']; ang_prim=r['ang_prim']; ang_scat=r['ang_scat']
eflu_prim=float(r['eflu_prim']); eflu_scat=float(r['eflu_scat']); N=int(r['N'])
int_main=r['int_main'].item() if hasattr(r['int_main'],'item') else r['int_main']
Vcyl=np.pi*1.0**2*0.2

i5=np.argmin(abs(depths-5.0))
prim_phi=0.5*(105.0/(100.0+Z))**2*(np.exp(-f_tot(E1)*Z)+np.exp(-f_tot(E2)*Z))
prim_mc=prim_tl[i5]/Vcyl/N
k=prim_phi/prim_mc
print("prim_phi/Phi0 at 5cm:",prim_phi)
print("prim_mc per hist:",prim_mc," k:",k)

scat_phi=scat_tl[i5]/Vcyl/N*k
print("scat_phi/Phi0 at 5cm:",scat_phi)
# spectra per Phi0
spec_prim_per=spec_prim/Vcyl/N*k
spec_scat_per=spec_scat/Vcyl/N*k
eflu_prim_per=eflu_prim/Vcyl/N*k
eflu_scat_per=eflu_scat/Vcyl/N*k
print("spec_prim sum:",spec_prim_per.sum()," spec_scat sum:",spec_scat_per.sum())
print("eflu_prim_per:",eflu_prim_per," eflu_scat_per:",eflu_scat_per)
print("mean scat E:",eflu_scat_per/spec_scat_per.sum())

# L1 quantities
L1_primary=prim_phi
L1_total=prim_phi+scat_phi
L1_scat_frac=scat_phi/L1_total
L1_scat_efrac=eflu_scat_per/(eflu_prim_per+eflu_scat_per)
# mean photon energy: total energy fluence / total fluence
E_bar=(eflu_prim_per+eflu_scat_per)/L1_total
print("L1:",L1_primary,L1_total,L1_scat_frac,L1_scat_efrac,E_bar)
# primary spectrum (lines)
spp=np.zeros(28)
spp[23]=0.5*np.exp(-f_tot(E1)*Z); spp[26]=0.5*np.exp(-f_tot(E2)*Z)
print("primary spectrum:", spp)
np.save('analysis_part1.npy', np.array([prim_phi,scat_phi,eflu_prim_per,eflu_scat_per,k],dtype=object), allow_pickle=True)