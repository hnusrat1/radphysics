import numpy as np
from scipy import integrate

# Check the problematic L5 calculations with proper units

SSD = 100.0
depth = 5.0
rho_water = 1.000
E_avg_co60 = 1.25285  # MeV

mu_rho_total = 0.05532  # cm^2/g
mu_compton = 0.04985
mu_rayleigh = 0.00298
mu_photoelectric = 0.00178
mu_pair = 0.00071

mu_tot = mu_compton + mu_rayleigh + mu_photoelectric + mu_pair

B_d = 1.45

# Energy transfer fractions per process
et_compton = 0.566
et_rayleigh = 0.0
et_photoelectric = 0.99
pair_E1_et = max(0, (1.1732 - 1.022)/1.1732)
pair_E2_et = max(0, (1.3325 - 1.022)/1.3325)
et_pair = (pair_E1_et + pair_E2_et)/2

# Mass energy-transfer coefficient
mu_rho_en = mu_compton*et_compton + mu_rayleigh*et_rayleigh + mu_photoelectric*et_photoelectric + mu_pair*et_pair

print(f"μ/ρ total = {mu_rho_total:.5f} cm²/g")
print(f"μ_en/ρ = {mu_rho_en:.5f} cm²/g")

# Attenuation
mu_linear = mu_rho_total * rho_water
attenuation_5cm = np.exp(-mu_linear * depth)

# Fluence at point relative to Phi0
L1_primary_per_phi0 = attenuation_5cm
L1_total_per_phi0 = attenuation_5cm * B_d

print(f"\nPrimary fluence/Φ₀ = {L1_primary_per_phi0:.4f}")
print(f"Total fluence/Φ₀ = {L1_total_per_phi0:.4f}")

# Kerma calculation - careful with units
# Kerma K [Gy] per unit Φ₀ [photons/cm² at reference]:
# K = Ψ × (μ_en/ρ) where Ψ is energy fluence [J/m²] or [MeV/cm² converted to Gy·m²]
# 
# More directly: K (J/kg) = Φ (photons/cm²) × E_photon (J/photon) × (μ_en/ρ) (cm²/g) × (1000 g/kg)
# But E in J needs conversion from MeV: 1 MeV = 1.60218×10⁻¹³ J

E_photon_J = E_avg_co60 * 1.60218e-13  # J/photon

# For K/Φ₀ at the scoring point:
# K/Φ₀ = (μ_en/ρ) × E_photon × conversion_factor_to_Gy_cm2
# Where Φ₀ is defined at 105 cm without phantom

# At the scoring point, photon fluence = Φ₀ × L1_total_per_phi0
# Energy fluence at point = Φ_point × E_avg (in MeV/cm² if Φ in photons/cm²)
# Convert to J/cm²: multiply by 1.60218e-13

energy_fluence_per_Phi0 = L1_total_per_phi0 * E_avg_co60  # MeV/cm² per incident photon at source
energy_fluence_J_per_Phi0 = energy_fluence_per_Phi0 * 1.60218e-13  # J/cm² per incident photon at source

# K (Gy) = energy_absorbed / mass = (energy_flux × absorption_coefficient) / density × path_length...
# Simplified for narrow beam: K = Ψ × (μ_en/ρ) where Ψ in J/m²
# Converting: 1 J/cm² = 10,000 J/m²

K_over_Phi0 = energy_fluence_J_per_Phi0 * 10000 * mu_rho_en * 1000  # ... this isn't right

# Let me do this more carefully:
# Φ₀ [photons/cm²] (at reference, no phantom)
# At depth d: Φ(d) = Φ₀ × exp(-μd) × B(d)
# Energy fluence Ψ(d) [J/m²] = Φ(d) [photons/m²] × E_photon [J]
#   = Φ₀[photons/cm²] × (10⁴ cm²/m²) × exp(-μd) × B(d) × E_photon[J]

# Kerma K[Gy] = Ψ(mass energy absorbed per unit volume) / ρ
# K = Ψ × (μ_en/ρ)
#   = [Φ₀ × 10⁴ × exp(-μd) × B(d) × E_photon(J)] × [(μ_en/ρ)(m²/kg)]

# (μ_en/ρ) in cm²/g = (μ_en/ρ) × 0.1 m²/kg
# So: K[Gy] = Φ₀[photons/cm²] × 10⁴[m²/cm²? No...] 

# Let's use base SI:
# Φ₀ has units: photons·cm⁻²  
# Convert to photons·m⁻²: multiply by 10⁴
# Ψ = Φ₀ × 10⁴ × exp(-μd) × B(d) × E_photon(J) in J·m⁻²
# μ_en/ρ in m²·kg⁻¹: convert from cm²·g⁻¹ by × 0.1
# K = Ψ × (μ_en/ρ) = [Φ₀×10⁴×exp×B×E] × [μ_en/ρ × 0.1] in (J/m²)×(m²/kg) = J/kg = Gy

K_per_Phi0_at_depth = (10**4) * attenuation_5cm * B_d * E_avg_co60 * 1.60218e-13 * mu_rho_en * 0.1

print(f"\nDirect K/Φ₀ calculation:")
print(f"  E_avg × e^-μd × B = {E_avg_co60 * attenuation_5cm * B_d:.4f} MeV·cm⁻² per Φ₀")
print(f"  In J/m²: {(E_avg_co60 * attenuation_5cm * B_d * 1.60218e-13):.4e} J/m² (per Φ₀)")
print(f"  Multiply by (μ_en/ρ)×0.1: {mu_rho_en * 0.1:.4e}")
print(f"  K/Φ₀ = {K_per_Phi0_at_depth:.6e} Gy·cm²")

# Alternatively using common radiation dosimetry conversion:
# 1 photon/cm² at 1 MeV → energy fluence = 1 MeV/cm² = 1.60218e-13 J/cm² = 1.60218e-9 J/m²
# With μ_en/ρ = 0.03011 cm²/g = 0.003011 m²/kg
# K = 1.60218e-9 × 0.003011 = 4.824e-12 Gy per photon/cm² per MeV
# Per MeV of energy fluence, not per photon!

# Correct approach:
# Φ₀ [photons/cm²] → energy fluence per MeV of photon energy = Φ₀ × E_avg [MeV·cm⁻²]
# K = Φ₀ × E_avg × (μ_en/ρ) × conversion
# Conversion: [MeV/cm²] → [Gy] requires:
# 1 MeV/cm² = 1.60218e-13 J/cm² = 1.60218e-9 J/m²  
# Divide by ρ in kg/m³ (= 1000 kg/m³ for water) gives J/(kg·m²)? No...

# Actually for slab geometry or point kernel:
# Differential energy deposition per unit length: dE/dx = Ψ × (μ_en)
# Integrated over thickness t: ΔE ≈ Ψ × (μ_en) × t
# Dose = ΔE/m = Ψ × (μ_en) × t / (ρ × A × t) = Ψ × μ_en/ρ per unit area considerations...

# Standard formula used in practice:
# K[Gy] = Ψ[J/m²] × (μ_en/ρ)[m²/kg]
#       = Ψ[MeV/cm²] × 1.60218e-9 [J/MeV·m²/cm²?] × (μ_en/ρ)×0.1 [m²/kg per cm²/g]

# Hmm, let me just use a simple practical conversion:
# Φ photons/cm² at 1 MeV → energy fluence 1 MeV/cm²
# K = 1 MeV/cm² × (μ_en/ρ)[cm²/g] × 1.60218e-13[J/MeV] × 1000[g/kg] = ... × 1.60218e-10 Gy

conversion_simple = 1.60218e-13 * 1000  # = 1.60218e-10

K_calc_simple = L1_total_per_phi0 * mu_rho_en * E_avg_co60 * conversion_simple
print(f"\nSimplified K/Φ₀ (Gy·cm²): {K_calc_simple:.6e}")

# These should give the same order of magnitude...
# Using direct SI conversion first:
K_SI = 10**4 * attenuation_5cm * B_d * E_avg_co60 * 1.60218e-13 * mu_rho_en * 0.1
print(f"Simplified K (check): {K_calc_simple:.6e}")
print(f"SI-based K (check): {K_SI:.6e}")

# They differ by factor of ~10⁴×0.1 = 10³ vs ×1000, so they match! Good.