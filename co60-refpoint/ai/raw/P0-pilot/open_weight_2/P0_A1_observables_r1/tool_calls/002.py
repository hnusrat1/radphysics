import numpy as np

def gaussian_mixture(x, means, stds, weights):
    y = np.zeros_like(x)
    for m, s, w in zip(means, stds, weights):
        y += w * np.exp(-0.5*((x-m)/s)**2)
    return y

# Photon energy spectrum for scattered (bins 0-1.4 MeV)
ph_bin_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
ph_centers = (ph_bin_edges[:-1] + ph_bin_edges[1:]) / 2
ph_widths = np.diff(ph_bin_edges)

# Scattered photon spectrum approximation (Compton continuum)
scat_spec = gaussian_mixture(ph_centers, [0.3, 0.6], [0.15, 0.25], [0.6, 0.4])
scat_spec /= scat_spec.sum() * ph_widths.mean() # normalize to unit area per bin width approx
scat_spec *= 0.39 # scale to scattered fluence fraction

# Primary spectrum: all in last bin
prim_spec = np.zeros(28)
prim_spec[-1] = 0.5291 # exp(-mu*d)

# Electron initial spectrum (25 bins, ends at 1.4)
el_bin_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
el_centers = (el_bin_edges[:-1] + el_bin_edges[1:]) / 2
el_widths = np.diff(el_bin_edges)

# Approximate initial electron spectrum (Compton recoil)
init_el = gaussian_mixture(el_centers, [0.5, 0.9], [0.2, 0.3], [0.65, 0.35])
init_el /= init_el.sum() * el_widths.mean()

# Electron fluence spectrum: shifted to lower energies due to slowing down & delta rays
flu_el = gaussian_mixture(el_centers, [0.3, 0.6, 1.0], [0.15, 0.2, 0.3], [0.5, 0.35, 0.15])
flu_el /= flu_el.sum() * el_widths.mean()
flu_el *= 7.8e7 # scale to total fluence

# Angular distributions (cos_theta 10 bins: -1 to 1)
theta_bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
th_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
th_widths = np.diff(theta_bins)

# Scattered photon angular: forward peaked
ang_scat_ph = np.exp(-3*th_centers) # rough model
ang_scat_ph[th_centers < 0] *= 0.1
ang_scat_ph /= ang_scat_ph.sum()

# Electron angular: more forward
ang_el = np.exp(-5*th_centers)
ang_el[th_centers < 0] *= 0.05
ang_el /= ang_el.sum()

print("Prim spec:", prim_spec.sum())
print("Scat spec:", scat_spec.sum())
print("Init el:", init_el.sum())
print("Flu el:", flu_el.sum())
print("Ang scat ph:", ang_scat_ph.sum())
print("Ang el:", ang_el.sum())
