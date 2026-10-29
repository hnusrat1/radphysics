import numpy as np
from scipy.integrate import quad

# Read data files
xcom = []
with open('/tmp/data/xcom_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            xcom.append([float(x) for x in line.split()])
xcom = np.array(xcom)

estar = []
with open('/tmp/data/estar_water.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
           estar.append([float(x) for x in line.split()])
estar = np.array(estar)

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

# Discretize cos(theta) uniformly
cos_vals = np.linspace(-1, 1, 20001)
w_KN = KN(cos_vals)
e_energies = e_KN(cos_vals)
d_cos = cos_vals[1] - cos_vals[0]

# Integrate properly using trapezoidal rule
total_int = np.sum(w_KN) * d_cos
print(f"Total integral (trapezoidal): {total_int:.6f}")

# Now bin
L3_comp_spec = np.zeros(n_ebins)
for i in range(n_ebins):
    mask = (e_energies >= ebins[i]) & (e_energies < ebins[i+1])
    bin_int = np.sum(w_KN[mask]) * d_cos
    L3_comp_spec[i] = bin_int / total_int

print(f"Spectrum sum: {np.sum(L3_comp_spec):.6f}")
print(f"First few values: {L3_comp_spec[:5]}")

# Check fraction in first bin [0.01, 0.05)
first_bin_mask = (e_energies >= ebins[0]) & (e_energies < ebins[1])
print(f"Fraction in first bin [0.01, 0.05): {L3_comp_spec[0]:.6f}")

# Mean energy from Compton
mean_E = np.sum(e_energies * w_KN * d_cos) / total_int
print(f"Mean Compton electron energy: {mean_E:.4f} MeV")