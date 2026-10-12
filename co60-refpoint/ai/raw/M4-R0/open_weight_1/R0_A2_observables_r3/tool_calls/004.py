import numpy as np
from scipy.interpolate import interp1d

# ---------- data ----------
d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_ph = d[:,0]; mu_coh=d[:,1]; mu_incoh=d[:,2]; mu_pe=d[:,3]; mu_pairN=d[:,4]; mu_pairE=d[:,5]; mu_tot=d[:,6]
f_coh = interp1d(E_ph, mu_coh, kind='linear', bounds_error=False, fill_value=(mu_coh[0], mu_coh[-1]))
f_incoh = interp1d(E_ph, mu_incoh, kind='linear', bounds_error=False, fill_value=(mu_incoh[0], mu_incoh[-1]))
f_pe = interp1d(E_ph, mu_pe, kind='linear', bounds_error=False, fill_value=(mu_pe[0], mu_pe[-1]))
f_pair = interp1d(E_ph, mu_pairN+mu_pairE, kind='linear', bounds_error=False, fill_value=(mu_pairN[0], mu_pairN[-1]))
f_tot = interp1d(E_ph, mu_tot, kind='linear', bounds_error=False, fill_value=(mu_tot[0], mu_tot[-1]))

d2 = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=d2[:,0]; Sc=d2[:,1]; Sr=d2[:,2]; St=d2[:,3]; R=d2[:,4]; Y=d2[:,5]; de=d2[:,6]
f_Sc = interp1d(Ee, Sc, kind='linear', bounds_error=False, fill_value=(Sc[0],Sc[-1]))
f_Sr = interp1d(Ee, Sr, kind='linear', bounds_error=False, fill_value=(Sr[0],Sr[-1]))
f_St = interp1d(Ee, St, kind='linear', bounds_error=False, fill_value=(St[0],St[-1]))
f_R  = interp1d(Ee, R, kind='linear', bounds_error=False, fill_value=(R[0],R[-1]))
f_Y  = interp1d(Ee, Y, kind='linear', bounds_error=False, fill_value=(Y[0],Y[-1]))

# analytic primary fluence per Phi0 at depth z (both lines)
E1=1.1732; E2=1.3325
def prim_phi_per_phi0(z):
    return 0.5*(105.0/(100.0+z))**2*(np.exp(-f_tot(E1)*z)+np.exp(-f_tot(E2)*z))
print("mu1,mu2,tot:", f_tot(E1), f_tot(E2))
print("prim fluence/Phi0 at z=5:", prim_phi_per_phi0(5.0))

# ---------- Compton helpers ----------
me=0.51099895
def kn_ratio(E, mu):
    alpha=E/me
    r=1.0/(1.0+alpha*(1.0-mu))
    return r*r*(r+1.0/r-(1.0-mu**2))   # unnormalized dsigma/dmu

def sample_compton_angle(E, size):
    # rejection sample mu in [-1,1]
    alpha=E/me
    maxf = 2.0 if alpha<10 else None
    # find rough max: maximize r^2(r+1/r-(1-mu^2)); just use 3.0 safe upper bound? Kn function <= ? at mu=1 =2; for small alpha maybe up to ~2? check
    out=np.empty(size)
    got=np.zeros(size,bool)
    tries=0
    # use vectorized rejection
    while not got.all():
        mu = np.random.uniform(-1,1,size)
        f = kn_ratio(np.full(size,E),mu)
        acc = np.random.uniform(0,3.0,size) < f
        sel = (~got)&acc
        got[sel]=True; out[sel]=mu[sel]
        tries+=1
        if tries>10000: 
            out[~got]=1.0; break
    return out

# test sample distribution vs known: average cos
mu = sample_compton_angle(1.25, 200000)
print("mean mu sampled:", mu.mean())
# expected for 1.25 MeV: average cos ~?