import numpy as np
from scipy.interpolate import interp1d

# Data already loaded above, continuing calculations

# Define bin edges as specified
photon_energy_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 
                      1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]  # 28 bins

electron_energy_bins = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 
                        1.05, 1.1, 1.15, 1.2, 1.4]  # 25 bins (last wider)

cos_theta_bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]  # 10 bins

# Cross sections computed earlier
E_co1 = 1.1732
E_co2 = 1.3325

# Total mass attenuation coefficients
xs_co1 = mu_over_rho['co1']
xs_co2 = mu_over_rho['co2']

# Fraction of interactions by type for Co-60 average
# Weighted by equal probability
def get_interaction_fractions(E):
    xs = mu_over_rho['co1'] if E == E_co1 else mu_over_rho['co2']
    
    total_without_rayleigh = xs['incoherent'] + xs['photoelectric'] + xs['pair_nuclear'] + xs['pair_electron']
    total_with_rayleigh = xs['total']  # includes coherent
    
    # Fractions of total interactions (including Rayleigh)
    f_compton = xs['incoherent'] / total_with_rayleigh if total_with_rayleigh > 0 else 0
    f_photo = xs['photoelectric'] / total_with_rayleigh if total_with_rayleigh > 0 else 0
    f_pair = (xs['pair_nuclear'] + xs['pair_electron']) / total_with_rayleigh if total_with_rayleigh > 0 else 0
    f_rayleigh = xs['coherent'] / total_with_rayleigh if total_with_rayleigh > 0 else 0
    
    return {
        'compton': f_compton,
        'photoelectric': f_photo,
        'pair': f_pair,
        'rayleigh': f_rayleigh,
        'total_cross_section': total_with_rayleigh,
        'total_without_rayleigh': total_without_rayleigh
    }

fractions_co1 = get_interaction_fractions(E_co1)
fractions_co2 = get_interaction_fractions(E_co2)

print("Interaction fractions:")
for name, fracs in [('Co-60 line 1', fractions_co1), ('Co-60 line 2', fractions_co2)]:
    print(f"{name}:")
    for k, v in fracs.items():
        print(f"  {k}: {v:.6f}")

# Average fractions (equal weight)
avg_fractions = {}
for key in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    avg_fractions[key] = 0.5 * fractions_co1[key] + 0.5 * fractions_co2[key]
    
print("\nAverage interaction fractions:")
for k, v in avg_fractions.items():
    print(f"  {k}: {v:.6f}")

# Energy transfer fractions (weighted by mean energy transferred)
# For Compton: <T>_Compton depends on angle, but roughly ~0.3-0.5 MeV for Co-60
# For photoelectric: almost full photon energy transferred
# For pair: ~E_gamma - 1.022 MeV (if above threshold)
# For Rayleigh: no energy transfer

# Let's use more detailed approach with Klein-Nishina for Compton