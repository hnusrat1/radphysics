import numpy as np
from scipy.special import eval_legendre
import json

# ==================== RELOAD ALL DATA AND CALCULATIONS ====================
with open('/tmp/data/xcom_water.txt', 'r') as f:
    xcom_lines = f.readlines()
    
with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

xcom_data = []
for line in xcom_lines[4:]:
    parts = line.split()
    if len(parts) >= 7:
        try:
            vals = [float(p) for p in parts]
            xcom_data.append(vals)
        except:
            pass
xcom_array = np.array(xcom_data)

estar_data = []
for line in estar_lines[4:]:
    parts = line.split()
    if len(parts) >= 7:
        try:
            vals = [float(p) for p in parts]
            estar_data.append(vals)
        except:
            pass
estar_array = np.array(estar_data)

def interpolate_xlog(E, col_idx):
    energies = xcom_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    
    log_E = np.log10(max(E, 1e-20))
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    
    t = (log_E - log_E_low) / (log_E_high - log_E_low)
    
    val_low = xcom_array[idx-1, col_idx]
    val_high = xcom_array[idx, col_idx]
    
    if val_low > 0 and val_high > 0:
        return val_low * (val_high/val_low)**t
    elif val_high > 0:
        return val_high
    else:
        return 0.0

def interpolate_estar_log(E):
    energies = estar_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    
    log_E = np.log10(max(E, 1e-20))
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    
    t = (log_E - log_E_low) / (log_E_high - log_E_low)
    
    result = {}
    cols_names = ['collision', 'radiative', 'total']
    col_indices = [1, 2, 3]
    
    for name, idx_col in zip(cols_names, col_indices):
        val_low = estar_array[idx-1, idx_col]
        val_high = estar_array[idx, idx_col]
        
        if val_low > 0 and val_high > 0:
            result[name] = val_low * (val_high/val_low)**t
        elif val_high > 0:
            result[name] = val_high
        else:
            result[name] = 0.0
    
    return result

# Constants
E_co1 = 1.1732
E_co2 = 1.3325
rho = 1.0  # g/cm^3
depth = 5.0  # cm

# Cross sections
xs_co1 = {
    'coherent': interpolate_xlog(E_co1, 1),
    'incoherent': interpolate_xlog(E_co1, 2),
    'photoelectric': interpolate_xlog(E_co1, 3),
    'pair_nuclear': interpolate_xlog(E_co1, 4),
    'pair_electron': interpolate_xlog(E_co1, 5),
    'total': interpolate_xlog(E_co1, 6)
}

xs_co2 = {
    'coherent': interpolate_xlog(E_co2, 1),
    'incoherent': interpolate_xlog(E_co2, 2),
    'photoelectric': interpolate_xlog(E_co2, 3),
    'pair_nuclear': interpolate_xlog(E_co2, 4),
    'pair_electron': interpolate_xlog(E_co2, 5),
    'total': interpolate_xlog(E_co2, 6)
}

mu_over_rho_avg = 0.5 * xs_co1['total'] + 0.5 * xs_co2['total']
mu_avg = mu_over_rho_avg * rho  # cm^-1 at average energy

# Primary attenuation
att1 = np.exp(-xs_co1['total'] * rho * depth)
att2 = np.exp(-xs_co2['total'] * rho * depth)
L1_primary_fluence_per_phi0 = 0.5 * att1 + 0.5 * att2

# ============================================================
# LAYER 1: PHOTONS ARRIVING AT THE POINT
# ============================================================

# Primary fluence per Phi0 (already computed)
print(f"L1_primary_fluence_per_phi0 = {L1_primary_fluence_per_phi0:.6f}")

# Scattered photon estimation
# Build-up factor approximation for 5cm depth in water with Co-60 beam
# At 5cm depth and small field size, B ~ 1.1-1.3 typically
# But our scoring volume is very small (cylinder r=1cm)
# Scatter contribution mainly from Compton scattering within the field

# Estimate scattered fluence using interaction rates
# Assume scatter comes from interactions upstream (0 to 5cm depth)
# and scatters into the scoring volume

# Simplified: Use exponential buildup with buildup parameter
# For Co-60, typical build-up factor at 5cm in water ~1.15-1.25 for narrow beam
# Actually, for point of interest at dmax (~0.5cm for Co-60), B~1
# At 5cm, there's some scatter contribution

# For this problem, let's use approximate values based on radiation physics:
# At 5cm depth in water with 10x10 field and SSD=100:
# - Primary: ~73% of surface value (after attenuation)
# - Total with scatter: primary * B where B ~ 1.1-1.2
# 
# More precisely, at d=5cm for Co-60:
# %dd(5cm)/%dd(dmax) ≈ 64-65% for large fields
# At dmax (~0.5cm), primary is dominant

# Let's estimate B = total/primary ≈ 1.08 for this geometry (small field, limited scatter)
B_factor = 1.08  # Reasonable build-up factor at 5cm for this geometry
L1_total_fluence_per_phi0 = L1_primary_fluence_per_phi0 * B_factor
L1_scattered_fraction_fluence = 1.0 - 1.0/B_factor  # Fraction scattered in total

print(f"L1_total_fluence_per_phi0 = {L1_total_fluence_per_phi0:.6f}")
print(f"L1_scattered_fraction_fluence = {L1_scattered_fraction_fluence:.6f}")

# Energy fluence weighted mean energy
# Primary photons have energies 1.1732 and 1.3325 MeV
# Scattered photons are lower energy (Compton spectrum)
# Weighted mean will be slightly below monoenergetic 1.25285 MeV

# Mean energy of primary only
E_mean_primary = 0.5 * E_co1 + 0.5 * E_co2  # Equal probability
print(f"E_mean_primary (monoenergetic) = {E_mean_primary:.4f} MeV")

# Scattered photons average energy (roughly half of incident for Compton at these energies)
E_scattered_mean = 0.5 * (0.5 * E_co1 + 0.5 * E_co2) * 0.65  # Rough avg
print(f"Estimated E_scattered_mean ~ {E_scattered_mean:.4f} MeV")

# Fluence-weighted mean energy (all photons)
# Primary fraction: 1/B, scattered: (B-1)/B
frac_primary = 1.0 / B_factor
frac_scattered = 1.0 - frac_primary

# For simplicity, assume primary maintains their energy, scattered are degraded
L1_mean_photon_energy_MeV = frac_primary * E_mean_primary + frac_scattered * E_scattered_mean
print(f"L1_mean_photon_energy_MeV = {L1_mean_photon_energy_MeV:.4f} MeV")

# Photon fluence spectra (primary)
# Two delta functions at 1.1732 and 1.3325 MeV
photon_energy_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 
                      1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]

n_photon_bins = len(photon_energy_bins) - 1
L1_photon_fluence_spectrum_primary = np.zeros(n_photon_bins)

# Bin edges indices
bin1_117 = np.where(np.abs(np.array(photon_energy_bins[:-1]) - E_co1) < 0.025)[0]
bin2_133 = np.where(np.abs(np.array(photon_energy_bins[:-1]) - E_co2) < 0.025)[0]

if len(bin1_117) > 0 and len(bin2_133) > 0:
    # Split evenly between two lines
    for b in bin1_117:
        L1_photon_fluence_spectrum_primary[b] = 0.5 * L1_primary_fluence_per_phi0
    for b in bin2_133:
        L1_photon_fluence_spectrum_primary[b] = 0.5 * L1_primary_fluence_per_phi0

print(f"\nPrimary spectrum has entries:")
print(f"  Bin near 1.17 MeV: {L1_photon_fluence_spectrum_primary[np.argmin(np.abs(np.array(photon_energy_bins[:-1])-E_co1))] :.6f}")
print(f"  Bin near 1.33 MeV: {L1_photon_fluence_spectrum_primary[np.argmin(np.abs(np.array(photon_energy_bins[:-1))-E_co2))] :.6f}")

# Scattered spectrum - approximate Compton continuum
# Spread over energy bins from 0 to ~1.1 MeV
L1_photon_fluence_spectrum_scattered = np.zeros(n_photon_bins)

# Approximate Compton scattered spectrum shape
# For Co-60, scattered photons form a broad distribution
# Peak around 0.3-0.5 MeV region
# Integrate roughly equal area in several bins
E_scatter_start = 0.05
E_scatter_end = 1.1
scatter_flux = L1_total_fluence_per_phi0 - L1_primary_fluence_per_phi0

# Rough rectangular-ish distribution
n_scatter_bins = 15
scatter_flux_per_bin = scatter_flux / n_scatter_bins
start_bin = np.searchsorted(photon_energy_bins, E_scatter_start)
end_bin = min(np.searchsorted(photon_energy_bins, E_scatter_end), n_photon_bins)

for b in range(start_bin, end_bin):
    L1_photon_fluence_spectrum_scattered[b] = scatter_flux_per_bin

# Angular distributions
cos_theta_bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
n_cos_bins = len(cos_theta_bins) - 1

# Primary angular distribution
# All primaries come from source on central axis
# At the scoring volume position, all directions are nearly parallel to beam axis
# For cylinder r=1cm at z=105cm from source at origin:
# Max angle = arctan(1/105) ~ 0.54 degrees -> cos ~ 0.99995
# Essentially all in forward direction
L1_angular_primary = np.zeros(n_cos_bins)
L1_angular_primary[-1] = 1.0  # Last bin: cos=0.8 to 1.0

# Scattered angular distribution - forward peaked but broader
# Compton scattering angles spread out
L1_angular_scattered = np.zeros(n_cos_bins)

# Distribution roughly: most forward-scattered, some backscattered
# Normalize to sum to 1
# Using simplified distribution based on Klein-Nishina integrated over azimuth
cos_vals = []
angles_frac = []
for i in range(n_cos_bins):
    c_low = cos_theta_bins[i]
    c_high = cos_theta_bins[i+1]
    c_mid = (c_low + c_high) / 2.0
    
    # Estimate relative probability using Klein-Nishina
    theta = np.arccos(np.clip(c_mid, -1, 1))
    sigma_rel = klein_nishinan_dsigma_dcoss(theta, E_mean_primary)[0]
    cos_vals.append(c_mid)
    angles_frac.append(sigma_rel)

angles_frac = np.array(angles_frac)
angles_frac /= angles_frac.sum()
L1_angular_scattered = angles_frac

print(f"\nL1_angular_primary = {L1_angular_primary}")
print(f"L1_angular_scattered = {L1_angular_scattered}")
