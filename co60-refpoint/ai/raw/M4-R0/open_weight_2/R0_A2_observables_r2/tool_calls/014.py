import numpy as np

# For L3: Electrons set in motion by photon interactions
# These are from Compton scatterings (dominant), plus tiny contributions from photoelectric and pair

# Maximum electron energy from Compton scattering:
def compton_max_energy(E_gamma):
    """Maximum kinetic energy transferred to electron in Compton scattering"""
    mc2 = 0.511  # MeV
    return E_gamma * 2 * E_gamma / (mc2 + 2 * E_gamma)

print("Maximum electron energies from Compton:")
for E in [1.1732, 1.3325]:
    Ee_max = compton_max_energy(E)
    print(f"  E_gamma={E:.4f} MeV -> E_e,max = {Ee_max:.4f} MeV")

# Average electron energy from Compton scattering (rough estimate ~0.25*E_gamma for high energies)
# Actually, more precisely it depends on the differential cross section

# For Co-60 average (~1.25 MeV), mean electron energy from Compton is roughly:
# From NIST ESTAR and analytical calculations, for 1.25 MeV photons in water:
# The mean energy transferred per Compton interaction is approximately 0.3-0.35 MeV

# Let me use a slightly better approximation based on Klein-Nishina
# Mean fractional energy transfer for Compton scattering at E~1.25 MeV is about 0.25

E_mean_gamma = 1.25  # approximate weighted average
mean_energy_transfer_compton = 0.25 * E_mean_gamma  # ~0.31 MeV

print(f"\nEstimated mean electron energy from Compton: {mean_energy_transfer_compton:.3f} MeV")

# For Layer 3 initial spectrum, I'll need to distribute electrons across energy bins
# The Compton electron spectrum has a characteristic shape - more low-energy electrons than high-energy
# Approximate distribution based on Klein-Nishina differential cross section:
# Flat in cos(theta_photon), which translates to specific electron energy distribution

# Let's create a rough initial spectrum for electrons
electron_bins = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                          0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])

bin_centers = 0.5 * (electron_bins[:-1] + electron_bins[1:])
bin_widths = np.diff(electron_bins)

# Rough spectrum assuming most electrons from Compton with mean energy ~0.31 MeV
# Use a Gaussian-like distribution centered around 0.3 MeV
initial_spectrum = np.zeros(len(bin_centers))
mean_E = 0.31
std_E = 0.15  # width of distribution

for i, E_center in enumerate(bin_centers):
    # Compton electron spectrum peaks at lower energies
    # Approximate with log-normal-like shape
    if E_center < 1.05:  # Only up to max possible Compton electron energy
        # Simplified spectrum shape
        prob = np.exp(-((np.log10(E_center) - np.log10(mean_E))/0.3)**2) if E_center > 0 else 0
        initial_spectrum[i] = prob

# Normalize
initial_spectrum = initial_spectrum / initial_spectrum.sum()

print("\nInitial electron spectrum (fraction per bin):")
for i, (E_low, E_high, frac) in enumerate(zip(electron_bins[:-1], electron_bins[1:], initial_spectrum)):
    print(f"  {E_low:.2f}-{E_high:.2f} MeV: {frac:.4f}")

# Check sum
print(f"Spectrum sum: {initial_spectrum.sum():.4f}")
