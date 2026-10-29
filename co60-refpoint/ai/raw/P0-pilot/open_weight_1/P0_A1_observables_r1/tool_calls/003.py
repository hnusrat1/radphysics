import numpy as np
import math

# Constants
mec2 = 0.511  # MeV
r0 = 2.8179403262e-13  # cm, classical electron radius
r0_sq = r0**2
NA = 6.02214076e23  # Avogadro's number

# Water properties
# H2O: molecular weight ~18.015 g/mol
# Electrons per molecule: 2*1 + 8 = 10
M_H2O = 18.01528  # g/mol
electrons_per_mol_H2O = 10 * NA  # electrons per mole
electrons_per_gram_H2O = electrons_per_mol_H2O / M_H2O
print(f"Electrons per gram of water: {electrons_per_gram_H2O:.4e}")

# Atoms per gram for water
atoms_H_per_gram = 2 * NA / M_H2O
atoms_O_per_gram = 1 * NA / M_H2O
print(f"H atoms per gram: {atoms_H_per_gram:.4e}")
print(f"O atoms per gram: {atoms_O_per_gram:.4e}")

# Klein-Nishina total cross section per electron
def klein_nishina_sigma(E_MeV):
    """Klein-Nishina total cross section per electron in cm^2"""
    alpha = E_MeV / mec2
    sigma = 2*np.pi*r0_sq * ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
                            + np.log(1+2*alpha)/(2*alpha) - (1+3*alpha)/(1+2*alpha)**2)
    return sigma

# Compton mass attenuation coefficient for water
def mu_Compton_rho(E_MeV):
    """Mass attenuation coefficient for Compton scattering in water (cm^2/g)"""
    sigma_e = klein_nishina_sigma(E_MeV)
    # per electron cross section times electrons per gram
    return sigma_e * electrons_per_gram_H2O

E0 = 2.0
sigma_e_2MeV = klein_nishina_sigma(E0)
mu_comp_rho_2MeV = mu_Compton_rho(E0)
print(f"Klein-Nishina σ per electron at 2 MeV: {sigma_e_2MeV:.4e} cm^2")
print(f"Compton μ/ρ at 2 MeV: {mu_comp_rho_2MeV:.6f} cm^2/g")

# Also compute the energy transfer for Compton
# Average energy of scattered photon and energy transferred to electron
# I'll integrate the differential cross section

# Compton scattering formula: E' = E / (1 + alpha*(1-cosθ))
# where alpha = E/mec^2

def compton_differential_cross_section(E, cos_theta):
    """Klein-Nishina differential cross section dσ/dΩ in cm^2/sr per electron"""
    alpha = E / mec2
    # E' = E / (1 + alpha*(1-cos_theta))
    ratio = 1.0 / (1.0 + alpha * (1.0 - cos_theta))
    E_prime = E * ratio
    # Klein-Nishina formula
    dsigma_dOmega = (r0_sq / 2.0) * ratio**2 * (ratio + 1.0/ratio - (1.0 - cos_theta**2))
    return dsigma_dOmega, E_prime

# Integrate to find average scattered photon energy and energy transfer
def compton_energy_transfer(E, n_theta=10000):
    """Compute average energy transferred to electron for Compton scattering"""
    cos_theta_vals = np.linspace(-1, 1, n_theta)
    dtheta = np.pi / (n_theta - 1)
    # dΩ = 2π * d(cosθ)
    # Actually, dΩ = 2π d(cos θ) = 2π sin θ dθ
    
    total_sigma = 0.0
    total_energy_transfer = 0.0
    total_energy_scattered = 0.0
    
    # Integrate over cosθ from -1 to 1
    # dΩ = 2π * d(cosθ)... wait, more precisely:
    # dΩ = sin θ dθ dφ = -d(cosθ) dφ
    # ∫ dΩ = ∫_{-1}^{1} ∫_{0}^{2π} dφ d(cosθ) = 2π ∫_{-1}^{1} d(cosθ)
    # So dΩ = 2π d(cosθ)
    
    for i in range(len(cos_theta_vals)):
        cosθ = cos_theta_vals[i]
        dsigma, E_prime = compton_differential_cross_section(E0, cosθ)
        # weight for integration: dΩ = 2π * d(cosθ)
        # Using trapezoidal integration
        if i == 0 or i == len(cos_theta_vals)-1:
            weight = 1.0
        else:
            weight = 2.0
        # Actually, for Simpson or trapezoidal, let me use a sinmpler approach
    # Let me use scipy integrate
    return None

print("\nTrying integration with scipy...")