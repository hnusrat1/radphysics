import numpy as np
import math

# Analyze the scatter spectrum
energies = [e for e, w in scat_data]
weights = [w for e, w in scat_data]

print(f"Min energy: {min(energies):.4f} MeV")
print(f"Max energy: {max(energies):.4f} MeV")
# Histogram
bins = np.linspace(0, 2, 21)
hist, edges = np.histogram(energies, bins=bins, weights=weights)
print("Energy bins (MeV) and fluence:")
for i in range(len(hist)):
    if hist[i] > 0:
        print(f"  {edges[i]:.2f}-{edges[i+1]:.2f}: {hist[i]:.4f}")

# Also check the interaction weights per energy
scat_weights2 = [w * mu_Total(e) * (math.pi * 1.0**2 * 0.2) for e, w in scat_data]
hist_int, _ = np.histogram(energies, bins=bins, weights=scat_weights2)
print("Interaction weight per bin:")
for i in range(len(hist_int)):
    if hist_int[i] > 0:
        print(f"  {edges[i]:.2f}-{edges[i+1]:.2f}: {hist_int[i]:.6f}")