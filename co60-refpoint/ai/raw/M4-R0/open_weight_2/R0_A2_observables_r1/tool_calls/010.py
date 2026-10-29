import numpy as np

# Load data
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
estar = np.loadtxt('/tmp/data/estar_water.txt')

# Interpolation functions
def get_xcom(E):
    # Returns dict of coeffs
    return {
        'rayleigh': np.interp(E, xcom[:,0], xcom[:,1]),
        'compton': np.interp(E, xcom[:,0], xcom[:,2]),
        'photo': np.interp(E, xcom[:,0], xcom[:,3]),
        'pair_nuc': np.interp(E, xcom[:,0], xcom[:,4]),
        'pair_el': np.interp(E, xcom[:,0], xcom[:,5]),
        'total': np.interp(E, xcom[:,0], xcom[:,6])
    }

c1 = get_xcom(1.1732)
c2 = get_xcom(1.3325)
c_avg = {k: (v1+v2)/2 for k,v1 in c1.items() for v2 in c2.items()}
# Correct avg computation
c_avg = {}
for k in c1:
    c_avg[k] = (c1[k] + c2[k]) / 2

mu_tot = c_avg['total']
mu_ray = c_avg['rayleigh']
mu_comp = c_avg['compton']
mu_pe = c_avg['photo']
mu_pair = c_avg['pair_nuc'] + c_avg['pair_el']

print("Mu tot:", mu_tot)

# Mu_tr/rho approximation: mu_en/rho for Co-60 in water is ~0.02979 cm^2/g.
mu_en = 0.02979 

d = 5.0
attenuation = np.exp(-mu_tot * d)
print("Primary fluence per Phi0:", attenuation)

# Buildup factor at 5 cm for Co-60 10x10 is ~1.16
B = 1.16
total_fluence = attenuation * B
scattered_frac_fluence = (B - 1) / B
print("Total fluence per Phi0:", total_fluence)
print("Scattered frac fluence:", scattered_frac_fluence)

# Mean energy: initial 1.25285 MeV. Scattered photons are lower energy.
# Approximate mean energy after 5 cm: ~1.15 MeV
mean_E_phot = 1.15

# Interaction fractions
total_mu = mu_tot
frac_ray = mu_ray / total_mu
frac_comp = mu_comp / total_mu
frac_pe = mu_pe / total_mu
frac_pair = mu_pair / total_mu
print("Interaction fracs:", {'rayleigh': frac_ray, 'compton': frac_comp, 'photo': frac_pe, 'pair': frac_pair})

# Energy transfer fractions: weighted by mu_tr_process.
# For Compton, mu_tr/mu ~ 0.45-0.5. For others ~1.0 (except Rayleigh ~0).
# Let's assume Compton transfers ~0.48 on average.
wt_comp = frac_comp * 0.48
wt_ray = 0.0
wt_pe = frac_pe * 1.0
wt_pair = frac_pair * 1.0
sum_wt = wt_comp + wt_ray + wt_pe + wt_pair
frac_et = {k: v/sum_wt for k,v in {'compton': wt_comp, 'rayleigh': wt_ray, 'photo': wt_pe, 'pair': wt_pair}.items()}
print("ET fracs:", frac_et)

# Kerma
K_per_Phi0 = attenuation * mu_en * 1.60217662e-13 # Gy cm^2 per photon/cm^2? 
# Wait, phi0 is in photons/cm^2. E is in MeV. 1 MeV = 1.602e-13 Gy kg / (kg/m^2) -> actually 1 MeV/g = 1.602e-10 Gy.
# So K[Gy] = Phi[cm^-2] * E[MeV] * (mu_en/rho)[cm^2/g] * 1.602e-10.
# But we need per Phi0. The prompt says "per unit Phi0". Usually this means if Phi0=1 cm^-2.
# Let's stick to numerical value without E factor? No, kerma depends on E.
# The prompt asks for "L5_K_over_phi0_Gy_cm2". This implies it's a number. I will compute it correctly.
# Actually, sometimes "Phi0" in these problems is normalized such that energy fluence Psi0 = Phi0 * 1 MeV/cm^2? 
# No, Phi0 is just fluence. I'll compute exact value.
K_val = attenuation * mu_en * 1.60217662e-10 # using E=1 MeV implicitly? Or should I multiply by mean E?
# Let's check typical units. If Phi0 is photons/cm^2, then K/Phi0 = (mu_en/rho) * <E> * 1.602e-10.
# I'll assume they want the numeric value for K/Phi0. I'll include <E>=1.25285.
E_mean_source = (1.1732 + 1.3325)/2
K_per_Phi0_exact = attenuation * mu_en * E_mean_source * 1.60217662e-10
print("K/Phi0:", K_per_Phi0_exact)
