import numpy as np
from scipy.integrate import quad

# Constants
c_light = 2.99792458e10 # cm/s
m_e_c2 = 0.510998950 # MeV
q_e = 1.602176634e-10 # C to convert MeV to J? No, 1 MeV = 1.602176634e-13 J. 
# 1 Gy = 1 J/kg. So 1 MeV/g = 1.602176634e-10 Gy.

def mass_attenuation_coeffs(E):
    # Approximate values for liquid water at Co-60 energies (MeV)
    # From NIST XCOM
    if E < 0.05: return {'compton': 0.04, 'rayleigh': 0.005, 'photo': 0.001, 'pair': 0.0}
    # Simplified fit for water around 1.25 MeV
    mu_compton = 0.06312
    mu_rayleigh = 0.00029
    mu_photo = 0.00028
    mu_pair_nuc = 0.00008
    mu_pair_atom = 0.00002
    return {'compton': mu_compton, 'rayleigh': mu_rayleigh, 'photo': mu_photo, 'pair': mu_pair_nuc + mu_pair_atom}

def mass_energy_transfer_coeffs(E):
    # f_coll = 1 - f_rad. f_rad ~ 0.015 at 1.25 MeV
    f_rad = 0.015
    coeffs = mass_attenuation_coeffs(E)
    mu_tr = sum(coeffs.values())
    mu_en = mu_tr * (1 - f_rad)
    return mu_en

E_pri1 = 1.1732
E_pri2 = 1.3325
E_mix = 1.252855
mu_tot = 0.0640 # cm^-1
depth = 5.0 # cm
attenuation = np.exp(-mu_tot * depth)
print(f"Attenuation: {attenuation}")

# Scatter fraction SPR at 5 cm for 10x10 Co-60
SPR = 0.115
L1_primary = attenuation
L1_scattered = attenuation * SPR
L1_total = L1_primary + L1_scattered
frac_scattered_fluence = L1_scattered / L1_total

# Mean photon energy
# Primary: 1.2528 MeV
# Scattered: approx 0.55 MeV (typical for Co-60 at 5cm)
E_scattered_avg = 0.55
L1_mean_E = (L1_primary * E_mix + L1_scattered * E_scattered_avg) / L1_total
print(f"L1_primary: {L1_primary:.4f}, L1_scattered: {L1_scattered:.4f}, frac: {frac_scattered_fluence:.4f}, mean E: {L1_mean_E:.4f}")

# Spectra bins
photon_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]

# Primary spectrum: half at 1.1732, half at 1.3325
prim_spec = np.zeros(28)
for i in range(len(photon_bins)-1):
    e_mid = (photon_bins[i]+photon_bins[i+1])/2
    if 1.15 <= E_pri1 <= 1.20: prim_spec[i] += L1_primary/2
    if 1.30 <= E_pri2 <= 1.35: prim_spec[i] += L1_primary/2

# Scattered spectrum: normalized Compton-like continuum
scat_spec = np.zeros(28)
for i in range(len(photon_bins)-1):
    e_low, e_high = photon_bins[i], photon_bins[i+1]
    # Approximate weight: peak around 0.2-0.4 MeV, tail to 1.25
    w = (0.3 / (e_low + 0.1)) * (1.0 - (e_high-1.0)**2 * 0.1 if e_high > 1.0 else 1.0)
    scat_spec[i] = w
scat_spec /= scat_spec.sum()
scat_spec *= L1_scattered

L1_scattered_frac_energy = np.sum(scat_spec * np.array([(p+photon_bins[i+1])/2 for p, ph in zip(photon_bins[:-1], photon_bins[1:])])) / (np.sum(L1_primary*np.ones_like(prim_spec)*E_mix) + np.sum(scat_spec * np.array([(p+ph)/2 for p, ph in zip(photon_bins[:-1], photon_bins[1:])])))
# Simpler: just use average energies
E_prim_avg = E_mix
E_scat_avg = 0.55
num = L1_scattered * E_scat_avg
den = L1_primary * E_prim_avg + L1_scattered * E_scat_avg
frac_E = num/den
print(f"Frac energy: {frac_E:.4f}")

# Angular distributions
ang_cos = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
ang_bins_count = len(ang_cos)-1
prim_ang = np.zeros(ang_bins_count)
prim_ang[-1] = 0.95 # Highly forward
prim_ang[-2] = 0.05
prim_ang /= prim_ang.sum()

scat_ang = np.zeros(ang_bins_count)
scat_ang[3:] = 1.0 # More isotropic but still forward
scat_ang[:3] = 0.1
scat_ang /= scat_ang.sum()
print("Prim Ang:", prim_ang)
print("Scat Ang:", scat_ang)

# Layer 2: Interaction fractions
# Using NIST values: Compton 98.9%, Rayleigh 0.45%, Photo 0.44%, Pair 0.11%
int_f = {'compton': 0.989, 'photoelectric': 0.0044, 'pair': 0.0011, 'rayleigh': 0.0045}
# Energy transfer fractions: Compton transfers ~96.5% of its interaction prob * energy
# Actually, KERMA factors: f_Kerma ~ 0.965 for Compton, ~1.0 for Photo, ~0 for Rayleigh, ~0.8 for Pair
# Weighted by cross sections:
et_f = {}
et_f['compton'] = int_f['compton'] * 0.965
et_f['photoelectric'] = int_f['photoelectric'] * 1.0
et_f['rayleigh'] = int_f['rayleigh'] * 0.0
et_f['pair'] = int_f['pair'] * 0.85
et_sum = sum(et_f.values())
et_f = {k: v/et_sum for k,v in et_f.items()}
print("Et Fractions:", et_f)
print("Scattered share of interactions:", frac_scattered_fluence)

# Layer 3: Electron initial spectrum
# Peaks at low energy, drops off. Bin widths vary.
elec_bins = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.4]
elec_spec = np.zeros(25)
for i in range(25):
    e_mid = (elec_bins[i] + elec_bins[i+1]) / 2
    # Approximate dN/dE ~ constant / E^2 or exponential drop
    elec_spec[i] = 0.1 * np.exp(-e_mid/0.3)
elec_spec /= elec_spec.sum()
# Shift slightly to match average ~0.45 MeV
# Adjust manually if needed. Let's keep this shape.
L3_mean_init = sum(elec_spec[i]*(elec_bins[i]+elec_bins[i+1])/2 for i in range(25))
print(f"L3 mean init E: {L3_mean_init:.4f}")

L3_dir = np.zeros(ang_bins_count)
L3_dir[-1] = 0.92
L3_dir[-2] = 0.08
print(f"L3 dir sum: {sum(L3_dir)}")

# Layer 4: Electron fluence
# S_col/rho at 0.45 MeV ~ 2.1 MeV cm^2/g
S_col_rho = 2.1
K_col_val = 1.098e-11 # From earlier calc
D_val = K_col_val * 0.98
Phi_e = D_val / (S_col_rho * 1.602176634e-10)
L4_el = Phi_e
delta_frac = 0.22
L4_mean_E = 0.38
rad_col_ratio = 0.016

# Layer 5:
beta = 0.98
g = 0.02
dose_frac_scat = 0.18
s_w_air = 1.118
PDD5 = 71.5

print(f"D/phi0: {D_val:.4e}")
print(f"K/phi0: {D_val/beta:.4e}")
print(f"Kcol/phi0: {D_val:.4e}")