import numpy as np

# Constants and setup
E_peaks = [1.1732, 1.3325]
bin_edges_ph = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
bin_edges_el = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
cos_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

# Bin centers
ph_centers = (bin_edges_ph[:-1] + bin_edges_ph[1:]) / 2
el_centers = (bin_edges_el[:-1] + bin_edges_el[1:]) / 2

# L1 Primary Fluence Spectrum
spec_prim = np.zeros(28)
for E in E_peaks:
    idx = np.searchsorted(bin_edges_ph[1:], E, side='left') - 1
    spec_prim[idx] += 0.5 # 50% each line relative to total primary fluence
# Normalize to represent relative probability, actual value handled by Phi0 scaling
print("Primary spectrum indices:", np.where(spec_prim > 0)[0], "values:", spec_prim[np.where(spec_prim > 0)])

# L1 Angular Primary: straight down beam axis -> all in last bin [0.8, 1]
ang_prim = np.zeros(10)
ang_prim[-1] = 1.0

# L1 Total Fluence & Scattered
mu_t = 0.0047 # cm^-1 approx for water @ 1.25 MeV
depth = 5.0
T = np.exp(-mu_t * depth) # ~0.9769
B_F = 1.045 # Fluence buildup factor estimated for shallow depth/scatter
spec_scat = np.zeros(28)
# Scatter spectrum: continuous, peaked at low E, tail to 1.17
# Approximate shape: f(E) ~ exp(-E/0.3) + uniform background
for i, E in enumerate(ph_centers):
    if E < 1.17:
        spec_scat[i] = np.exp(-E/0.35) + 0.1 * np.random.rand() # dummy shape, will normalize later
        
# We will finalize numbers manually after running calculations
print("Setup complete.")