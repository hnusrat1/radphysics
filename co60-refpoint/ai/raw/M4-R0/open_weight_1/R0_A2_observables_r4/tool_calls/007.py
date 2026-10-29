import numpy as np, time

# ---------- constants ----------
mec2 = 0.5109989461  # MeV
r_e2 = 0.07940615    # (classical electron radius)^2 in barns? no: r_e^2/2 in cm^2: 2.494e-25 cm^2 per electron... use KN in terms of sigma
# We'll use KN differential normalized to the XCOM incoherent cross section for sampling.

# photon cross-section data
ph = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
Eph = ph[:,0]; coh=ph[:,1]; inc=ph[:,2]; pe=ph[:,3]; pn=ph[:,4]; pel=ph[:,5]; tot=ph[:,6]
def mklog(x,y):
    ok=(y>0)&(x>0)
    return np.interp # placeholder
from scipy.interpolate import interp1d
fcoh = interp1d(np.log(Eph),np.log(coh),kind='linear',bounds_error=False,fill_value='extrapolate')
finc = interp1d(np.log(Eph),np.log(inc),kind='linear',bounds_error=False,fill_value='extrapolate')
fpe  = interp1d(np.log(Eph),np.log(pe),kind='linear',bounds_error=False,fill_value='extrapolate')
fpn  = interp1d(np.log(Eph),np.log(pn),kind='linear',bounds_error=False,fill_value='extrapolate')
fpel = interp1d(np.log(Eph),np.log(pel),kind='linear',bounds_error=False,fill_value='extrapolate')
ftot = interp1d(np.log(Eph),np.log(tot),kind='linear',bounds_error=False,fill_value='extrapolate')
def mu_coh(E): return float(np.exp(fcoh(np.log(float(E)))))
def mu_inc(E): return float(np.exp(finc(np.log(float(E)))))
def mu_pe(E):  return float(np.exp(fpe(np.log(float(E)))))
def mu_pn(E):  return float(np.exp(fpn(np.log(float(E)))))
def mu_pel(E): return float(np.exp(fpel(np.log(float(E)))))
def mu_tot(E): return float(np.exp(ftot(np.log(float(E)))))

# check values
E1,E2=1.1732,1.3325
print("mu_tot E1,E2:",mu_tot(E1),mu_tot(E2))
print("exp(-mu*5):",np.exp(-mu_tot(E1)*5),np.exp(-mu_tot(E2)*5))

# solid angle of pyramid
from scipy.integrate import dblquad
a=0.05
Omega = dblquad(lambda v,u: 1.0/(1+u*u+v*v)**1.5, -a, a, lambda u:-a, lambda u:a)[0]
print("Omega_pyr =",Omega, " Omega*105^2 =",Omega*105*105)
print("flux per source photon at axis r=105 (1/(Omega*105^2)) =",1/(Omega*105*105))