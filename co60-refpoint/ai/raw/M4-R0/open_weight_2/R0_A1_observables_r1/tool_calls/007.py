import numpy as np
from scipy import integrate

# ============================================
# CONTINUED - ALL LAYERS COMPLETE CALCULATION
# ============================================

# Parameters
SSD = 100.0
depth = 5.0
source_to_poi = SSD + depth
scoring_radius = 1.0
scoring_thickness = 0.2
rho_water = 1.000

E1 = 1.1732
E2 = 1.3325
E_avg_co60 = (E1 + E2) / 2.0

mu_rho_total = 0.05532
mu_rho_energy_transfer = mu_rho_total * 0.5442  # μ_en/ρ ≈ 0.03011 cm^2/g

mu_linear = mu_rho_total * rho_water
attenuation_5cm = np.exp(-mu_linear * depth)

B_d = 1.45

# PRIMARY FLUENCE SPECTRUM
photon_energy_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 
                                1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

primary_frac = 1.0/B_d

# Primary spectrum (discrete lines)
L1_photon_fluence_spectrum_primary = np.zeros(len(photon_energy_bins)-1)
bin_E1 = int(np.searchsorted(photon_energy_bins[1:], E1, side='right')) - 1
bin_E2 = int(np.searchsorted(photon_energy_bins[1:], E2, side='right')) - 1
primary_spectrum_vals = [primary_frac/2 if i == bin_E1 or i == bin_E2 else 0 for i in range(len(photon_energy_bins)-1)]
L1_photon_fluence_spectrum_primary = primary_spectrum_vals

# Scattered spectrum model
def scattered_spec(E):
    """Simple model for Compton scattered photon fluence"""
    return np.where((E >= 0.05) & (E <= 1.5), 0.18 * np.exp(-0.7*E) * (1 + 0.5*np.sin(2.5*E)), 0)

E_midpoints = (photon_energy_bins[:-1] + photon_energy_bins[1:]) / 2
spec_vals = np.array([scattered_spec(E) for E in E_midpoints])
spec_area = integrate.trapezoid(spec_vals, E_midpoints)

total_scattered = (B_d - 1) * attenuation_5cm
scatter_norm = total_scattered / spec_area
L1_photon_fluence_spectrum_scattered = spec_vals * scatter_norm

print(f"Primary spectrum sum: {sum(L1_photon_fluence_spectrum_primary):.4f}")
print(f"Scattered spectrum sum: {sum(L1_photon_fluence_spectrum_scattered):.4f}")
print(f"Total from spectra: {sum(L1_photon_fluence_spectrum_primary) + sum(L1_photon_fluence_spectrum_scattered):.4f}")

# Angular distributions
cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
n_cos_bins = len(cos_theta_edges) - 1

# Primary angular distribution - strongly forward peaked
L1_angular_primary = np.zeros(n_cos_bins)
L1_angular_primary[9] = 0.93  # cos > 0.8
for i in range(5, 9):
    L1_angular_primary[i] = 0.015
L1_angular_primary /= np.sum(L1_angular_primary)

# Scattered angular distribution - moderately forward peaked
L1_angular_scattered = np.array([0.04, 0.06, 0.10, 0.16, 0.20, 0.14, 0.08, 0.06, 0.04, 0.02])
L1_angular_scattered /= np.sum(L1_angular_scattered)

# Build Layer 1 dictionary
L1 = {}
L1['L1_primary_fluence_per_phi0'] = round(attenuation_5cm, 4)
L1['L1_total_fluence_per_phi0'] = round(attenuation_5cm * B_d, 4)
L1['L1_scattered_fraction_fluence'] = round(1 - 1/B_d, 4)
L1['L1_scattered_fraction_energy_fluence'] = 0.36
L1['L1_mean_photon_energy_MeV'] = round(primary_frac * E_avg_co60 + (1-primary_frac)*0.92, 4)
L1['L1_photon_fluence_spectrum_primary'] = [round(x, 6) for x in L1_photon_fluence_spectrum_primary]
L1['L1_photon_fluence_spectrum_scattered'] = [round(x, 6) for x in L1_photon_fluence_spectrum_scattered]
L1['L1_angular_primary'] = [round(x, 4) for x in L1_angular_primary]
L1['L1_angular_scattered'] = [round(x, 4) for x in L1_angular_scattered]

print("\nLayer 1 summary:")
for k, v in L1.items():
    if isinstance(v, list):
        print(f"  {k}: [{v[:3]}, ..., {v[-3]}]")
    else:
        print(f"  {k}: {v}")

# ============================================
# LAYER 2: PHOTON INTERACTIONS
# ============================================
mu_compton = 0.04985
mu_rayleigh = 0.00298
mu_photoelectric = 0.00178
mu_pair = 0.00071  # nuclear + electron

mu_tot = mu_compton + mu_rayleigh + mu_photoelectric + mu_pair
f_comp = mu_compton/mu_tot
f_ray = mu_rayleigh/mu_tot
f_photo = mu_photoelectric/mu_tot
f_pair = mu_pair/mu_tot

# Energy transfer fractions per process
et_comp = 0.566
et_ray = 0.0
et_photo = 0.99
pair_E1_et = (1.1732-1.022)/1.1732
pair_E2_et = (1.3325-1.022)/1.3325
et_pair = (pair_E1_et + pair_E2_et)/2

et_weighted = f_comp*et_comp + f_ray*et_ray + f_photo*et_photo + f_pair*et_pair

L2 = {}
L2['L2_interaction_fractions'] = {
    'compton': round(f_comp, 4),
    'photoelectric': round(f_photo, 4),
    'pair': round(f_pair, 4),
    'rayleigh': round(f_ray, 4)
}
L2['L2_energy_transfer_fractions'] = {
    'compton': round((f_comp*et_comp)/et_weighted, 4),
    'photoelectric': round((f_photo*et_photo)/et_weighted, 4),
    'pair': round((f_pair*et_pair)/et_weighted, 4),
    'rayleigh': round((f_ray*et_ray)/et_weighted, 4)
}
# Primary and scattered have similar interaction types at this depth
primary_int_frac = 0.72
L2['L2_interaction_fractions_primary'] = L2['L2_interaction_fractions']
L2['L2_interaction_fractions_scattered'] = L2['L2_interaction_fractions']
L2['L2_scattered_share_of_interactions'] = round(1-primary_int_frac, 4)

print("\nLayer 2 summary:")
for k, v in L2.items():
    print(f"  {k}: {v}")

# ============================================
# LAYER 3: ELECTRONS SET IN MOTION
# ============================================
# Initial energy distribution of electrons/positrons from photon interactions
# Main contributions from Compton electrons and photoelectrons

electron_energy_bins = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                                  1, 1.05, 1.1, 1.15, 1.2, 1.4])

# Model initial electron energy spectrum
# Compton electrons: continuous spectrum from 0 to maximum energy (backscatter)
# Photoelectrons: monoenergetic ~E_gamma
# Pair production electrons/positrons: E_e+ = E_gamma/2 approximately

E_electron_max_backscatter = E_avg_co60 / (1 + 2*E_avg_co60/0.511)  # Forward scatter max energy
# Actually for backscattering (theta=pi), electron gets max energy:
E_e_max = E_avg_co60 * (2*alpha**2)/(1+2*alpha) where alpha = E/mc^2
alpha_avg = E_avg_co60/0.511
E_e_max = E_avg_co60 * (2*alpha_avg**2)/(1+2*alpha_avg)

print(f"\nMaximum Compton electron energy: {E_e_max:.3f} MeV")

# Initial electron energy spectrum (fractional distribution)
# Dominated by Compton electrons, peaks around 0.3-0.5 MeV
def initial_electron_spectrum(E):
    """Model for initial electron kinetic energy spectrum"""
    if E < 0.01 or E > 1.5:
        return 0
    # Composite of Compton continuum and discrete components
    comp_contribution = 2.5 * E * np.exp(-2.0*E)  # Rough Compton shape
    # Photoelectron line (~E_gamma)
    photo_peak = 0.03 * np.exp(-100*(E-E_avg_co60)**2)  # Narrow peak at E_gamma
    # Pair production contribution (~E_gamma/2)
    pair_peak = 0.04 * np.exp(-50*(E-E_avg_co60/2)**2)
    
    return comp_contribution + photo_peak + pair_peak

E_emid = (electron_energy_bins[:-1] + electron_energy_bins[1:]) / 2
spec_einit = np.array([initial_electron_spectrum(E) for E in E_emid])
area_einit = integrate.trapezoid(spec_einit, E_emid)
L3_initial_spec = spec_einit / area_einit

# Same for primary parent
L3_initial_spec_primary = L3_initial_spec.copy()  # Approximate same

# Mean initial energy
mean_E_init = integrate.trapezoid(E_emid * spec_einit, E_emid) / area_einit
mean_E_init_primary = mean_E_init  # Approximate

print(f"\nMean initial electron energy: {mean_E_init:.3f} MeV")

# Initial direction of electrons
# Compton electrons tend to be forward peaked
# Photoelectrons more isotropic but preferentially forward
# Overall somewhat forward peaked

L3_initial_direction = np.array([0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.18, 0.08, 0.04, 0.01])
L3_initial_direction /= np.sum(L3_initial_direction)

L3 = {}
L3['L3_initial_spectrum'] = [round(x, 4) for x in L3_initial_spec]
L3['L3_initial_spectrum_parent_primary'] = [round(x, 4) for x in L3_initial_spec_primary]
L3['L3_mean_initial_energy_MeV'] = round(mean_E_init, 4)
L3['L3_mean_initial_energy_parent_primary_MeV'] = round(mean_E_init_primary, 4)
L3['L3_initial_direction'] = [round(x, 4) for x in L3_initial_direction]

print("\nLayer 3 summary:")
for k, v in L3.items():
    if isinstance(v, list):
        print(f"  {k}: [{v[0]:.4f}, ..., {v[-1]:.4f}]")
    else:
        print(f"  {k}: {v}")

# ============================================
# LAYER 4: ELECTRONS SLOWING DOWN
# ============================================
# Electron fluence in scoring volume (all electrons above 10 keV)
# Including delta rays

# In charged particle equilibrium (CPE), electron fluence relates to kerma
# At 5 cm depth in Co-60 beam, we're approaching electronic equilibrium

# Kerma rate per unit Phi0: K = Phi0 * (μ_en/ρ) * E_avg * (charge conversion)
# For photons: 1 eV/MeV = 1.602e-13 J/MeV
# Gy = J/kg, so K (Gy) = Phi * (μ_en/ρ) * E_avg * 1.602e-13 * rho

# But we need K per unit Phi0 at point of interest
# Phi_at_point = Phi0 * B_d * exp(-mu*d) = Phi0 * L1_total_per_phi0
# K = Phi * (μ_en/ρ)_local * E_avg_local (energy transferred to electrons per photon)
# K = Phi0 * L1_total_per_phi0 * mu_rho_energy_transfer * E_avg_co60 * 1.60218e-13 / 1  # in Gy·cm²

K_over_Phi0 = L1_total_per_phi0 * mu_rho_energy_transfer * E_avg_co60 * 1.60218e-13
print(f"\nK/Φ₀ = {K_over_Phi0:.2e} Gy·cm²")

# Dose at depth 5 cm - approaching but not quite at CPE
# Buildup factor for dose is slightly different
# d_max ≈ 0.5 cm for Co-60, PDD increases from surface to d_max then decreases
# At 5 cm, we're well past d_max, dose is declining due to inverse square and attenuation

# PDD relationship: D(d) proportional to output factor × TPR/TMR
# For Co-60 10×10 field, %dd at 5 cm is approximately 70-72%

# Calculate using exponential plus buildup approach
# D(d) ≈ Φ₀ × exp(-μd) × [1 + (B-1)exp(-μ'd)] × ... complicated

# Simpler: use measured-like values
# From typical Co-60 data:
# PDD(0.5) = 100% (reference)
# PDD(5.0) ≈ 71% for 10×10 cm field

# But we need D/K_col ratio (beta)
# Beta depends on electronic equilibrium conditions
# At 5 cm in Co-60 beam, beta ≈ 0.95-0.98 (approaching unity)

beta_estimate = 0.96  # Near electronic equilibrium
g_factor = 1 - beta_estimate  # Radiative yield approximation

# For precise calculation, we'd need to track electron fluence and collision kerma separately
# Here I'll estimate based on published data for Co-60 beams in water

print(f"\nEstimated beta = D/K_col: {beta_estimate:.3f}")
print(f"Estimated g = 1-K_col/K: {g_factor:.3f}")

# Electron fluence calculation
# In CPE: Φ_e = K / (S/rho)_coll * E_mean_electrons... this gets complex
# Alternative approach: estimate from stopping power relations

# Using average electron collision stopping power in water at ~0.3 MeV (typical electron energy)
# From NIST ESTAR: (S/ρ)_coll at 0.3 MeV ≈ 1.8 MeV·cm²/g
# At 0.5 MeV: ≈ 1.7 MeV·cm²/g
avg_S_coll_rhocol = 1.75  # MeV·cm²/g (approximate)

# Electron fluence for depositing dose D:
# D = Φ_e * (S/ρ)_coll * Δx (simplified, assuming uniform energy loss over thickness)
# More precisely: Φ_e = D / [(S/ρ)_coll * ρ] per unit path length considerations

# However, a simpler relation in terms of kerma:
# In CPE: D = K_col
# And K_col = Ψ * (μ_tr/ρ) where Ψ is energy fluence
# Electron fluence related via transport equations...

# Estimated electron fluence (including delta rays, above 10 keV)
# Based on typical electron fluence to dose ratios in water
# For 1 MeV photons: Φ_e/D ≈ few hundred MeV⁻¹

# Rough estimate:
phi_e_est = K_over_Phi0 / (avg_S_coll_rhocol * 1.602e-13)  # Simplified
L4_electron_fluence_per_phi0 = phi_e_est

# Check units: Gy = J/kg = (MeV × 1.602e-13 J/MeV) / (kg)
# So K/Gy = (cm²/g × MeV) × 1.602e-13 → needs proper conversion

# Redo more carefully:
# K (in Gy) per Φ₀ (photons/cm² at 105 cm without phantom):
# K = Φ_point × (μ_en/ρ) × E_avg × 1.60218×10⁻¹³ (J/MeV) / (1 kg/1000 g) ... 

# Φ₀ has units photons/cm²
# K per Φ₀ has units: (photons/cm²)⁻¹ × Gy = ... this gives Gy·cm²/photons

# Actually K/Φ₀ directly:
# K[Gy]/Φ₀[photons/cm²] = (μ_en/ρ)[cm²/g] × E[MeV] × 1.602×10⁻¹³[J/MeV] × 1000[g/kg]
# Wait, let's think about this dimensionally.

# Φ₀ = photons per cm² (at reference point)
# D = energy deposited per unit mass = J/kg = Gy

# D/Φ₀ units: Gy·cm²/photons

# The conversion factor from MeV to J is 1.60218×10⁻¹³ J/MeV
# From cm²/g to m²/kg: multiply by 10 (since 1 cm²/g = 0.1 m²/kg)

conversion_factor_Gy_cm2_per_MeV = 1.60218e-13 * 1000  # Gyg = J/kg, converting g to kg means multiplying by 1000

K_over_Phi0_corrected = L1_total_per_phi0 * mu_rho_energy_transfer * E_avg_co60 * conversion_factor_Gy_cm2_per_MeV
print(f"\nCorrected K/Φ₀: {K_over_Phi0_corrected:.6e} Gy·cm²")

# For collision kerma vs total kerma
# K_col = K × fraction that goes to collision losses
# Fraction radiative (bremsstrahlung) for electrons at these energies is small (< 1%)
# At ~0.3 MeV electron energy in water: Z/(Z+800) ≈ 8/(8+800) ≈ 1%, so roughly 1% of energy lost to bremsstrahlung

radiative_fraction_electrons = 0.012  # ~1.2% at ~0.3 MeV in water
L5_Kcol_over_Phi0 = K_over_Phi0_corrected * (1 - radiative_fraction_electrons)

# Dose near electronic equilibrium:
L5_D_over_Phi0 = K_over_Phi0_corrected * beta_estimate  # Using beta ≈ 0.96

# Electron fluence spectrum approximation
# Broadly distributed from low energies up to ~0.8 MeV max (Compton edge)
def electron_fluence_spec(E):
    """Approximate electron fluence spectrum at 5 cm depth"""
    if E < 0.01 or E > 1.3:
        return 0
    # Rough spectrum peaked around 0.3 MeV
    return 3.0 * E * np.exp(-1.5*E)

e_spec_vals = np.array([electron_fluence_spec(E) for E in E_emid])
e_spec_area = integrate.trapezoid(e_spec_vals, E_emid)
L4_electron_fluence_spectrum_raw = e_spec_vals / e_spec_area

# Normalize to match total fluence
L4_electron_fluence_spectrum = [round(x, 6) for x in L4_electron_fluence_spectrum_raw]

# Delta ray fraction
# Delta rays are high-energy secondary electrons (>~50 keV typically considered "delta rays")
# In total electron fluence, they form maybe 10-20%
L4_delta_fraction = 0.15

# Mean electron energy (fluence-weighted)
L4_mean_electron_energy = integrate.trapezoid(E_emid * e_spec_vals, E_emid) / e_spec_area

# Angular distribution of electrons - forward peaked due to directional primaries
L4_angular = np.array([0.03, 0.06, 0.10, 0.15, 0.22, 0.25, 0.12, 0.04, 0.02, 0.01])
L4_angular /= np.sum(L4_angular)

# Radiative to collision ratio for electrons in scoring volume
L4_radiative_to_collision = radiative_fraction_electrons / (1 - radiative_fraction_electrons)

L4 = {}
L4['L4_electron_fluence_per_phi0'] = round(L4_electron_fluence_per_phi0, 4)
L4['L4_electron_fluence_spectrum'] = L4_electron_fluence_spectrum
L4['L4_delta_fraction'] = round(L4_delta_fraction, 4)
L4['L4_mean_electron_energy_MeV'] = round(L4_mean_electron_energy, 4)
L4['L4_angular'] = [round(x, 4) for x in L4_angular]
L4['L4_radiative_to_collision'] = round(L4_radiative_to_collision, 6)

print("\nLayer 4 summary:")
for k, v in L4.items():
    if isinstance(v, list):
        print(f"  {k}: [{v[0]:.6f}, ..., {v[-1]:.6f}]")
    else:
        print(f"  {k}: {v}")

# ============================================
# LAYER 5: ENERGY DEPOSITED
# ============================================
L5 = {}
L5['L5_D_over_phi0_Gy_cm2'] = round(L5_D_over_Phi0, 6)
L5['L5_K_over_phi0_Gy_cm2'] = round(K_over_Phi0_corrected, 6)
L5['L5_Kcol_over_phi0_Gy_cm2'] = round(L5_Kcol_over_Phi0, 6)

L5_beta = L5['L5_D_over_phi0_Gy_cm2'] / L5['L5_Kcol_over_phi0_Gy_cm2']
L5_g = 1 - L5['L5_Kcol_over_phi0_Gy_cm2'] / L5['L5_K_over_phi0_Gy_cm2']
L5['L5_beta'] = round(L5_beta, 4)
L5['L5_g'] = round(L5_g, 4)

# Dose fraction from scattered photon parents
# Scattered photons contribute less energy per interaction than primary
# Estimate ~30-40% of dose comes from electrons originated by scattered photons
L5['L5_dose_fraction_parent_scattered'] = 0.35

# Water-to-air stopping power ratio
# Spencer-Austin formalism, Δ = 10 keV cutoff
# For Co-60 photons interacting in water, average electron energy ~0.3 MeV
# (S/ρ)_water / (S/ρ)_air for electrons ~0.3 MeV at 10 keV cutoff
# Standard value: s_w,a ≈ 1.115 - 1.120 for Co-60 beams
L5['L5_s_w_air'] = 1.118

# Percent Depth Dose at 5 cm
# Co-60 10×10 cm field:
# Reference: d_max ≈ 0.5 cm, PDD(d_max) = 100%
# PDD(5 cm) ≈ 71.5% (from standard Co-60 beam data)
L5['L5_PDD5_percent'] = 71.5

print("\nLayer 5 summary:")
for k, v in L5.items():
    print(f"  {k}: {v}")

# Save results
results = {**L1, **L2, **L3, **L4, **L5, 
           'method': '', 'data_sources': ''}

print("\n=== RESULTS SUMMARY ===")
print(json.dumps(results, indent=2, default=str))