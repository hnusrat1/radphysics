import numpy as np

def gen_spectrum(bins, centers, widths, means, stds, weights, scale=1.0):
    y = np.zeros(len(centers))
    for m, s, w in zip(means, stds, weights):
        y += w * np.exp(-0.5*((centers-m)/s)**2)
    # normalize per bin
    y /= (y.sum() * widths.mean())
    return y * scale

ph_bin_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
ph_centers = (ph_bin_edges[:-1] + ph_bin_edges[1:]) / 2
ph_widths = np.diff(ph_bin_edges)

el_bin_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
el_centers = (el_bin_edges[:-1] + el_bin_edges[1:]) / 2
el_widths = np.diff(el_bin_edges)

theta_bins = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
th_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
th_widths = np.diff(theta_bins)

# L1 primary spectrum: all in last bin
L1_prim_spec = np.zeros(28)
L1_prim_spec[-1] = 0.5291

# L1 scattered spectrum
L1_scat_spec = gen_spectrum(ph_centers, ph_widths, [0.3, 0.6], [0.15, 0.25], [0.6, 0.4], scale=0.39)

# L2 fractions
L2_int = {"compton": 0.963, "photoelectric": 0.002, "pair": 0.0, "rayleigh": 0.035}
L2_et = {"compton": 0.998, "photoelectric": 0.002, "pair": 0.0, "rayleigh": 0.0}
L2_int_primary = L2_int.copy()
L2_int_scattered = {"compton": 0.970, "photoelectric": 0.003, "pair": 0.0, "rayleigh": 0.027}
L2_scat_share = 0.40

# L3 initial electron spectrum
L3_init_spec = gen_spectrum(el_centers, el_widths, [0.55, 0.9], [0.2, 0.3], [0.7, 0.3], scale=1.0)
L3_init_spec_pr = gen_spectrum(el_centers, el_widths, [0.6, 1.0], [0.18, 0.25], [0.6, 0.4], scale=1.0)
L3_mean_E = 0.76
L3_mean_E_pr = 0.79

# L3 angular
L3_ang = np.exp(-4*th_centers)
L3_ang[th_centers < 0] *= 0.08
L3_ang /= L3_ang.sum()

# L4 electron fluence spectrum
L4_el_spec = gen_spectrum(el_centers, el_widths, [0.25, 0.55, 0.95], [0.15, 0.2, 0.25], [0.5, 0.35, 0.15], scale=7.8e7)
L4_delta_frac = 0.18
L4_mean_E = 0.82
L4_ang = np.exp(-6*th_centers)
L4_ang[th_centers < 0] *= 0.03
L4_ang /= L4_ang.sum()
L4_T_C = 0.003

# L5
L5_D_p = 0.0249
L5_K_p = 0.0259
L5_Kc_p = 0.0249
L5_beta = 0.96
L5_g = 0.04
L5_dose_scatt = 0.38
L5_sw_air = 1.116
L5_PDD5 = 88.5

print("L1_prim:", L1_prim_spec[-1])
print("L1_scat sum:", np.trapz(L1_scat_spec, ph_centers))
print("L3_init sum:", np.trapz(L3_init_spec, el_centers))
print("L3_init_pr sum:", np.trapz(L3_init_spec_pr, el_centers))
print("L4_el sum:", np.trapz(L4_el_spec, el_centers))
print("L3_ang sum:", L3_ang.sum())
print("L4_ang sum:", L4_ang.sum())
