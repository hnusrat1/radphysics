import numpy as np

def gen_spectrum(centers, widths, means, stds, weights, scale=1.0):
    y = np.zeros(len(centers))
    for m, s, w in zip(means, stds, weights):
        y += w * np.exp(-0.5*((centers-m)/s)**2)
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

# L1 spectra
L1_prim_spec = np.zeros(28)
L1_prim_spec[-1] = 0.5291
L1_scat_spec = gen_spectrum(ph_centers, ph_widths, [0.3, 0.6], [0.15, 0.25], [0.6, 0.4], scale=0.39)

# L3 spectra
L3_init_spec = gen_spectrum(el_centers, el_widths, [0.55, 0.9], [0.2, 0.3], [0.7, 0.3])
L3_init_spec_pr = gen_spectrum(el_centers, el_widths, [0.6, 1.0], [0.18, 0.25], [0.6, 0.4])

# L4 spectrum
L4_el_spec = gen_spectrum(el_centers, el_widths, [0.25, 0.55, 0.95], [0.15, 0.2, 0.25], [0.5, 0.35, 0.15], scale=7.8e7)

# Angular
L3_ang = np.exp(-4*th_centers); L3_ang[th_centers < 0] *= 0.08; L3_ang /= L3_ang.sum()
L4_ang = np.exp(-6*th_centers); L4_ang[th_centers < 0] *= 0.03; L4_ang /= L4_ang.sum()
L1_angular_scattered = np.exp(-3*th_centers); L1_angular_scattered[th_centers < 0] *= 0.1; L1_angular_scattered /= L1_angular_scattered.sum()
L1_angular_primary = np.zeros(10); L1_angular_primary[-1] = 1.0

print("Check sums:")
print("L1_prim:", L1_prim_spec.sum(), "trapz:", np.trapz(L1_prim_spec, ph_centers))
print("L1_scat trapz:", np.trapz(L1_scat_spec, ph_centers))
print("L3_init sum:", L3_init_spec.sum())
print("L4_el sum:", L4_el_spec.sum())
print("Angular sums:", L3_ang.sum(), L4_ang.sum(), L1_angular_scattered.sum())
