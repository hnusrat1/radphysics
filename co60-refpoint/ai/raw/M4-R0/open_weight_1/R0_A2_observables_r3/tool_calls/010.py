import numpy as np
from scipy.interpolate import interp1d

d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
_Em=d[:,0]; _mu_incoh=d[:,2]; _mu_tot=d[:,6]
f_incoh=interp1d(_Em,_mu_incoh,bounds_error=False,fill_value=(_mu_incoh[0],_mu_incoh[-1]))
f_tot=interp1d(_Em,_mu_tot,bounds_error=False,fill_value=(_mu_tot[0],_mu_tot[-1]))

me=0.51099895
P=np.array([0.,0.,5.0])
src=np.array([0.,0.,-100.0])
rng=np.random.default_rng(42)

def kn_ratio(E,mu):
    alpha=E/me; r=1/(1+alpha*(1-mu))
    return r*r*(r+1/r-(1-mu**2))

def kn_total_scaled(E):
    # integrate 2pi * kn_ratio over mu to get relative total; use scipy
    from scipy.integrate import quad
    val,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu), -1,1)
    return val

def compton_Ep(E,mu):
    alpha=E/me
    return E/(1+alpha*(1-mu))

# single scatter contribution at P per history, 10x10 field, E=1.25 averaged (use lines)
E_lines=[(1.1732,0.5),(1.3325,0.5)]
Nvol=4000000
# sample uniform in phantom box
xs=rng.uniform(-15,15,Nvol); ys=rng.uniform(-15,15,Nvol); zs=rng.uniform(0,30,Nvol)
tot=0.0
for E,frac in E_lines:
    # cone half width at depth z: 5*(1+z/100)
    hw=5*(1+zs/100.0)
    inbeam=(np.abs(xs)<hw)&(np.abs(ys)<hw)
    # primary fluence per history
    D=np.sqrt(xs**2+ys**2+(zs+100)**2)
    Phi0=np.where(inbeam, 100.0/D**2 * np.exp(-f_tot(E)*(D-100.0)), 0.0)
    # direction source->r
    drx=xs/D; dry=ys/D; drz=(zs+100)/D
    # direction r->P
    rx=P[0]-xs; ry=P[1]-ys; rz=P[2]-zs
    R=np.sqrt(rx**2+ry**2+rz**2)
    urx=rx/R; ury=ry/R; urz=rz/R
    cosT = drx*urx+dry*ury+drz*urz
    Ep=compton_Ep(E,cosT)
    # dsigma/dOmega / sigma : kn_ratio / (kn total/2pi?) kn_ratio = dsigma/dmu = 2pi dsigma/dOmega
    dsdOm_by_sigma = kn_ratio(E,cosT)/kn_total_scaled(E)  # since kn_ratio = 2pi dσ/dΩ
    # attenuation of scattered photon
    att=np.exp(-f_tot(Ep)*R)
    contrib = Phi0* f_incoh(E)* dsdOm_by_sigma*att/R**2
    tot += frac*np.sum(contrib)
# volume integral: dV = 30^3 / Nvol
dV=30**3/Nvol
Phi_s1 = tot*dV
print("single-scatter fluence per history at P:", Phi_s1)
print("MC scattered fluence per history (from before):", 0.00409)
print("primary fluence per history at P:", 100.0/105.0**2*np.exp(-f_tot(1.25)*5))