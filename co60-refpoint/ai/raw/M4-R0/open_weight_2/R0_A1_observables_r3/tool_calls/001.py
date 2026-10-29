import numpy as np

# Define bin edges
photon_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
electron_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
cos_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

print("Photon bins:", len(photon_edges)-1)
print("Electron bins:", len(electron_edges)-1)
print("Cos bins:", len(cos_edges)-1)

# Generate plausible scattered photon fluence spectrum (normalized to sum to ~0.087)
x_p = (photon_edges[:-1] + photon_edges[1:])/2
spec_scatt = np.exp(-x_p/0.25) * 0.5
spec_scatt /= spec_scatt.sum() * 0.0872

# Generate plausible electron initial spectrum
x_e = (electron_edges[:-1] + electron_edges[1:])/2
spec_e_init = x_e**0.5 * np.exp(-x_e/0.3)
spec_e_init /= spec_e_init.sum()

# Angular scattered
ang_scatt = np.array([0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.15, 0.12])
ang_scatt /= ang_scatt.sum()

# Angular scattered photons similarly
ang_ph_scatt = np.copy(ang_scatt)

# Electron initial direction
ang_e_init = np.array([0.01, 0.02, 0.04, 0.08, 0.12, 0.15, 0.18, 0.20, 0.14, 0.06])
ang_e_init /= ang_e_init.sum()

# Electron fluence spectrum
spec_e_fluence = spec_e_init * 0.7 # mostly same shape but scaled
delta_part = np.zeros_like(spec_e_fluence)
delta_part[2:10] = 0.05 # delta rays in mid-low energy
spec_e_fluence += delta_part
spec_e_fluence /= spec_e_fluence.sum()

# Electron angular
ang_e = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.312])
ang_e /= ang_e.sum()

print("Spec scatt sum:", spec_scatt.sum())
print("Spec e init sum:", spec_e_init.sum())
print("Ang scatt sum:", ang_scatt.sum())
print("Spec e fluence sum:", spec_e_fluence.sum())
print("Ang e sum:", ang_e.sum())