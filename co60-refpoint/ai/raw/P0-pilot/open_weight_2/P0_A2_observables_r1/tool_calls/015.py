import numpy as np

# Parameters
E_photon = 2.0
alpha = E_photon / 0.511

def KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    s = 1.0 - ct**2
    return (1.0/k**2)*(k + 1.0/k - s)

def e_KN(ct):
    k = 1.0 + alpha*(1.0-ct)
    return E_photon*(1 - 1.0/k)

n_ebins = 25
ebins = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                  1.1, 1.15, 1.2, 1.4])

cos_vals = np.linspace(-1, 1, 100001)
w_KN = KN(cos_vals)
e_energies = e_KN(cos_vals)
d_cos = cos_vals[1] - cos_vals[0]
total_int = np.sum(w_KN) * d_cos

# Check distribution of electron energies
print(f"Min e energy: {np.min(e_energies):.4f} MeV")
print(f"Max e energy: {np.max(e_energies):.4f} MeV")

# Fraction in different ranges
mask_low = e_energies < ebins[0]
mask_high = e_energies >= ebins[-1]
mask_mid = ~mask_low & ~mask_high

frac_low = np.sum(w_KN[mask_low]) * d_cos / total_int
frac_high = np.sum(w_KN[mask_high]) * d_cos / total_int
frac_mid = np.sum(w_KN[mask_mid]) * d_cos / total_int

print(f"\nFraction analysis:")
print(f"Below 0.01 MeV: {frac_low:.6f}")
print(f"Above 1.4 MeV: {frac_high:.6f}")
print(f"In [0.01, 1.4] MeV: {frac_mid:.6f}")
print(f"Sum: {frac_low + frac_high + frac_mid:.6f}")

# Find cos values corresponding to E = 0.01 and E = 1.4
for target_E in [0.01, 1.4]:
    ct_idx = np.argmin(np.abs(e_energies - target_E))
    print(f"E = {target_E} MeV corresponds to cos(theta) = {cos_vals[ct_idx]:.6f}")

# So electrons with high energies (> 1.4 MeV) correspond to cos(theta) close to -1
# Electrons with low energies (< 0.01 MeV) correspond to cos(theta) close to 1

# For proper L3 spectrum normalized over given bins:
L3_comp_spec_raw = np.zeros(n_ebins)
for i in range(n_ebins):
    mask_i = (e_energies >= ebins[i]) & (e_energies < ebins[i+1])
    L3_comp_spec_raw[i] = np.sum(w_KN[mask_i]) * d_cos / total_int

# Normalize to sum to 1 over these bins
L3_comp_spec_normalized = L3_comp_spec_raw / np.sum(L3_comp_spec_raw)

print(f"\nNormalized Compton spectrum sum: {np.sum(L3_comp_spec_normalized):.6f}")
print(f"L3_comp_spec_normalized: {L3_comp_spec_normalized[:5]}...{L3_comp_spec_normalized[-5:]}")