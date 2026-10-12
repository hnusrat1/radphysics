import numpy as np

# Cross-section fractions at Co-60 weighted energies
mu_coherent = 3.691167e-05  # cm^2/g (Rayleigh)
mu_compton = 6.309348e-02    # cm^2/g (incoherent)
mu_photo = 2.390013e-06      # cm^2/g (photoelectric) 
mu_pair_nuc = 2.903899e-05   # cm^2/g (pair nuclear)
mu_total_with_coh = 6.316173e-02  # cm^2/g (total with coherent)
rho = 1.0  # g/cm^3

# Fractions of interactions by process
fraction_rayleigh = mu_coherent / mu_total_with_coh
fraction_compton = mu_compton / mu_total_with_coh  
fraction_photo = mu_photo / mu_total_with_coh
fraction_pair_nuc = mu_pair_nuc / mu_total_with_coh
fraction_pair_ele = 0  # no pair production in electron field at these energies

print("Interaction fractions by process:")
print(f"  Rayleigh (coherent): {fraction_rayleigh:.6e}")
print(f"  Compton (incoherent): {fraction_compton:.6f}")
print(f"  Photoelectric: {fraction_photo:.6e}")
print(f"  Pair production (nuclear): {fraction_pair_nuc:.6e}")
print(f"  Total check: {fraction_rayleigh + fraction_compton + fraction_photo + fraction_pair_nuc:.6f}")

# For L2_interaction_fractions (all photons):
L2_interactions = {
    'compton': fraction_compton,
    'photoelectric': fraction_photo,
    'pair': fraction_pair_nuc,  # summing nuclear and electron fields
    'rayleigh': fraction_rayleigh
}
print("\nL2 interaction fractions:", L2_interactions)
