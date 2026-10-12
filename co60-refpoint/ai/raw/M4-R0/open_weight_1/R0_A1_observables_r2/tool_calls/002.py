
import numpy as np
from scipy.integrate import quad

# constants
r_e = 2.8179403262e-13  # cm
r_e2 = r_e*r_e
mec2 = 0.5109989461
NA = 6.02214076e23

# water: n_e per cm3
rho_w = 1.0
M_w = 18.01528  # g/mol
n_mol = NA/M_w  # molecules per cm3 (since 1 cm3 = 1 g)
n_e_w = 10*n_mol  # electrons per cm3

print("n_e water =", n_e_w)

# ---------- Compton (Klein-Nishina) ----------
def kn_dsig_dOm(E, costh):
    k = E/mec2
    Ep = E/(1.0 + k*(1.0-costh))
    ratio = Ep/E
    bracket = 1.0/ratio + ratio - (1.0 - costh*costh)
    return 0.5*r_e2*ratio*ratio*bracket

def kn_total(E):
    f = lambda th: kn_dsig_dOm(E, np.cos(th))*np.sin(th)
    v,_ = quad(f, 0, np.pi, epsabs=1e-30, limit=400)
    return 2*np.pi*v

def mu_compton(E):
    return n_e_w*kn_total(E)

# ---------- coherent (Rayleigh) using independent atom Gaussian form factors ----------
# form factor F(q) = Z*exp(-q^2 * a^2 / 2), q in (MeV/c)*?  use q = (E/hc)*sqrt(2(1-cos))
# hc = 197.3269804 MeV*fm ; convert to 1/angstrom: q(1/A) = E(MeV)*sqrt(2(1-costh))/(hc_A) with hc_A = 197.32698 MeV*1e-5 A? 
# Actually hbar*c = 197.3269804 MeV*fm = 1973.269804e-5 MeV*Angstrom = 0.0197327 MeV*Angstrom
# q in 1/A = E(MeV)*(2 sin(th/2))/(0.0197327 MeV*A)
hcA = 0.01973269804  # MeV*Angstrom

# atomic params for O and H: screening radii guesses
a_O = 0.30  # Angstrom-ish scale
a_H = 0.25
Z_O = 8.0; Z_H = 1.0

def ray_dsig_dOm(E, costh):
    q = E*np.sqrt(2*(1-costh))/hcA  # 1/A
    FO = Z_O*np.exp(-(q*q*a_O*a_O)/2.0)
    FH = Z_H*np.exp(-(q*q*a_H*a_H)/2.0)
    F2 = FO*FO + 2.0*FH*FH   # per molecule (2 H)
    return 0.5*r_e2*(1.0+costh*costh)*F2

def mu_coherent(E):
    f = lambda th: ray_dsig_dOm(E, np.cos(th))*np.sin(th)
    v,_ = quad(f, 0, np.pi, epsabs=1e-30, limit=400)
    return n_mol*2*np.pi*v

# ---------- photoelectric, empirical ----------
def mu_photo(E):
    # rough: mu/rho = A*E^-3.3, tuned so mu_pe ~0.002 at 0.1 MeV
    A = 0.002*(0.1)**3.3
    return A*E**(-3.3)

# ---------- pair ----------
def mu_pair(E):
    if E <= 1.022: return 0.0
    return 0.0045*(E-1.022)

def mu_total(E):
    return mu_compton(E)+mu_coherent(E)+mu_photo(E)+mu_pair(E)

for E in [0.01,0.05,0.1,0.2,0.5,1.0,1.1732,1.25,1.3325,1.5,2.0]:
    print(f"E={E:.4f} MeV: comp={mu_compton(E):.5f} coh={mu_coherent(E):.5f} pho={mu_photo(E):.5f} pair={mu_pair(E):.5f} tot={mu_total(E):.5f} cm2/g")

# mu_en (collision kerma coefficient) approximate table for water
# smooth parameterization: use given points
Epts = np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0,3.0])
muen = np.array([4.79,1.23,0.541,0.1537,0.0682,0.0409,0.0320,0.0260,0.0255,0.0275,0.0268,0.0288,0.0295,0.0328,0.0329,0.0322,0.0310,0.0303,0.0297,0.0294,0.0284,0.0266,0.0238])
from scipy.interpolate import interp1d
muen_interp = interp1d(Epts, muen, kind='cubic', bounds_error=False, fill_value='extrapolate')
for E in [0.05,0.1,0.3,0.5,0.8,1.0,1.1732,1.25,1.3325,1.5]:
    print("mu_en", E, muen_interp(E))

# stopping powers Bethe for water and air
def bethe_col(E, Z_A, I, rho, dens_corr=0.0):
    # E kinetic MeV
    gamma = 1.0 + E/mec2
    beta2 = 1.0 - 1.0/(gamma*gamma)
    Tmax = E  # MeV, for electron max transfer ~E (approx)
    # use MEAN EXCITATION: ln(2mc2 beta2 gamma2 / I) - beta2 - delta
    arg = 2*mec2*beta2*gamma2/(I*1e-6)  # I in MeV
    L = np.log(arg) - beta2 - dens_corr
    return 0.153536*(Z_A)*(L)/beta2  # MeV cm2/g

# water: Z_A = 10/18.0153 = 0.5551, I=78 eV
def Scol_w(E):
    return bethe_col(E, 10/18.01528, 78.0, 1.0)
# air: Z_A ~ 7.3/14.7? standard air: Z_eff/A ~0.4992, I=85.7 eV
def Scol_a(E):
    return bethe_col(E, 0.4992, 85.7, 0.001205)

def Srad_w(E):
    # radiative: approx, tune so g ~ 0.004 at 1.25
    return E*0.0062*0.001205/0.001205*1.0  # placeholder tune
    # we'll calibrate below

print("\nStopping powers water:")
for E in [0.01,0.05,0.1,0.5,1.0,1.25,1.4]:
    print(E, Scol_w(E), "air", Scol_a(E))
