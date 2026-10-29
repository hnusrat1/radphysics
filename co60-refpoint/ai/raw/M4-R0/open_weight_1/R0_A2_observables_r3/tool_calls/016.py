import importlib.util, os, numpy as np
from scipy.interpolate import interp1d
spec=importlib.util.spec_from_file_location("mc_module", os.path.join(os.getcwd(),"mc_module.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

# data for mu_en
d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
_Em=d[:,0]; _incoh=d[:,2]; _pe=d[:,3]; _pair=d[:,4]+d[:,5]; _tot=d[:,6]
f_incoh=interp1d(_Em,_incoh,bounds_error=False,fill_value=(_incoh[0],_incoh[-1]))
f_pe=interp1d(_Em,_pe,bounds_error=False,fill_value=(_pe[0],_pe[-1]))
f_pair=interp1d(_Em,_pair,bounds_error=False,fill_value=(_pair[0],_pair[-1]))
f_tot=interp1d(_Em,_tot,bounds_error=False,fill_value=(_tot[0],_tot[-1]))

# ESTAR for radiative yield g(E)
d2=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
_Ee=d2[:,0]; _Y=d2[:,5]
f_Y=interp1d(_Ee,_Y,bounds_error=False,fill_value=(_Y[0],_Y[-1]))

me=0.51099895
def kn_ratio(E,mu):
    a=E/me; r=1/(1+a*(1-mu)); return r*r*(r+1/r-(1-mu**2))
from scipy.integrate import quad
def av_electron_E(E):
    # average Compton electron energy
    num,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu)*(E - E/(1+(E/me)*(1-mu))),-1,1)
    den,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu),-1,1)
    return num/den
def mu_tr(E):
    Eg = E/me
    # compton transfer fraction
    fC = av_electron_E(E)/E
    mt = f_incoh(E)*fC + f_pe(E)*1.0 + f_pair(E)*(1-1.022/E if E>1.022 else 0)
    return mt
def mu_en(E):
    mt=mu_tr(E)
    # radiative fraction approx: for the electrons produced, use g at average electron energy of the dominant compton
    # simple: use g(E) weighted: for compton electrons mean Te = av_electron_E; for PE Te=E; pair Te=(E-1.022)/2 each
    # total g = sum_x mu_x * g_x / mu_tr
    Ec=av_electron_E(E)
    gc = f_Y(Ec) if Ec>0 else 0
    gc = f_Y(max(Ec,0.01))
    gpe = f_Y(E) if E>0 else 0
    gpair = f_Y(max((E-1.022)/2,0.01)) if E>1.022 else 0
    gbar = (f_incoh(E)*fC*gc + f_pe(E)*gpe + f_pair(E)*(1-1.022/E)*gpair)/max(mt,1e-30)
    return mt*(1-gbar)

E1=1.1732; E2=1.3325
mue1=mu_en(E1); mue2=mu_en(E2); mut1=mu_tr(E1); mut2=mu_tr(E2)
print("mu_en 1.1732,1.3325:",mue1,mue2,"mu_tr:",mut1,mut2)

depths=np.array([0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0,15.0,20.0,25.0,29.0])
res=mod.run_photon_mc(N=600000,depths=depths,seed=123, HW=15,ZD=30)
print("time",res['time'],"iters",res['iters'])
V=np.pi*0.2
prim_tl=res['prim_tl']; scat_tl=res['scat_tl']
# primary fluence per Phi0 analytic
prim_phi = 0.5*(105.0/(100+depths))**2*(np.exp(-f_tot(E1)*depths)+np.exp(-f_tot(E2)*depths))
prim_mc = prim_tl/V/res['N']
k = prim_phi/prim_mc
scat_phi = scat_tl/V/res['N']*k
print("k (cm2/hist) at each depth:", np.round(k,3))
# scattered mean energy from main cylinder (5cm)
es5 = res['eflu_scat']/max(res['scat_tl'][np.argmin(abs(depths-5))],1e-12)
print("mean scat E at 5cm:", es5)
# Kcol per Phi0: primary
C=1.602e-10
Kcol_p = prim_phi*C*(0.5*(mue1*E1+mue2*E2))  # careful: prim_phi = sum of lines; each line 0.5*... so weighted average mu_en*E = (mue1*E1+mue2*E2)/2
print("check: Kcol_p at 5cm per Phi0 Gy cm2:", Kcol_p[np.argmin(abs(depths-5))])
# scattered: use mean E_s and mu_en at mean (approx across depths, use 0.45)
Es_avg = 0.45
Kcol_s = scat_phi*C*mu_en(Es_avg)*Es_avg
Kcol_t = Kcol_p+Kcol_s
# approximate dose ~ Kcol for z>=0.5
dz = depths
it=0
print("  z     Kcol_p/Phi0   Kcol_s/Phi0   Kcol_t/Phi0   PDD%")
for i,z in enumerate(depths):
    print(f"{z:5.1f}  {Kcol_p[i]:.4e}  {Kcol_s[i]:.4e}  {Kcol_t[i]:.4e}  {100*Kcol_t[i]/np.max(Kcol_t):6.2f}")
imax=np.argmax(Kcol_t)
print("dmax index:", depths[imax], "PDD5=%", 100*Kcol_t[np.argmin(abs(depths-5))]/Kcol_t[imax])
print("PDD at 10:", 100*Kcol_t[np.argmin(abs(depths-10))]/Kcol_t[imax])