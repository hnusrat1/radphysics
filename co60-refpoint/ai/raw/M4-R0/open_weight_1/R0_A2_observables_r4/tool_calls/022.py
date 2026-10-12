import numpy as np
from scipy.integrate import quad

el=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
Ee=el[:,0]; Sc=el[:,1]; Sr=el[:,2]; St=el[:,3]; R=el[:,4]; Y=el[:,5]
mec2=0.5109989461
r0=2.8179403262e-13  # cm
NA=6.02214076e23
ZA_water = 0.55509  # electrons/g * ? actually N_A*Z/A per gram = 6.022e23*0.55509 = 3.343e23 /g
Ne_per_g = NA*ZA_water
print("Ne per g:",Ne_per_g)

def moler_prefactor(E):
    gamma = 1.0 + E/mec2
    beta2 = 1.0 - 1.0/(gamma*gamma)
    return 2*np.pi*r0*r0*mec2/beta2  # cm^2*MeV

def dsigma_dT(E,T):
    # Møller, per target electron, cm^2/MeV. T <= E/2
    gamma = 1.0 + E/mec2
    A = moler_prefactor(E)
    # bracket with units 1/MeV^2
    br = 1.0/T**2 + 1.0/(E-T)**2 - (2*gamma-1)/(gamma*gamma) * 1.0/(T*(E-T))
    return A*br

# check total stopping power integral consistency: S_col = Ne_per_g * int_0^{E/2} T dσ  with appropriate lower cutoff
# We'll compute restricted S (delta>Delta) and see it's small vs S_col
Delta=0.010
def S_delta(E):
    # energy in knock-ons > Delta, MeV cm^2/g
    val,_=quad(lambda T: T*dsigma_dT(E,T), Delta, E/2)
    return Ne_per_g*val
for E in [0.2,0.5,1.0,1.3325]:
    sd=S_delta(E)
    sc=float(np.interp(E,Ee,Sc))
    print(f"E={E}: S_col(file)={sc:.4f} S_delta>10keV={sd:.4f} restricted={sc-sd:.4f} frac_delta={sd/sc:.3f}")

# also number of delta above cutoff per cm: Ne_per_g*rho(=1)*int dσ
def N_delta_cm(E):
    val,_=quad(lambda T: dsigma_dT(E,T), Delta, E/2)
    return Ne_per_g*val  # per cm (rho=1)
for E in [0.2,0.5,1.0]:
    print(f"E={E}: N_delta per cm = {N_delta_cm(E):.3f}")

# Verify Møller momentum/angle formula
def cos_delta(T,E):
    return np.sqrt(T*(E+2*mec2)/(E*(T+2*mec2)))
print("cos delta T=E/2,E=1:",cos_delta(0.5,1.0), "theta=",np.degrees(np.arccos(cos_delta(0.5,1.0))))