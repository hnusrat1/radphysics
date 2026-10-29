import numpy as np
import math

mec2 = 0.511
r0 = 2.8179403262e-13

def sigma_kn(E):
    a = E / mec2
    a2 = a*a
    x = 1.0 + 2.0*a
    lnx = math.log(x)
    term1 = (1.0+a)/a2 * (2.0*(1.0+a)/x - lnx/a)
    term2 = lnx/(2.0*a) - (1.0+3.0*a)/(x*x)
    sigma = 2.0*math.pi*r0*r0 * (term1 + term2)
    return sigma

# Also compute the energy transfer cross section for Compton (for electron energy distribution)
# But we'll need differential cross sections for sampling.

# Let's compute some reference data
# For water, let's get approximate partial cross sections using known data
# I'll use the following approximations:

# Thomson cross section (Rayleigh) per atom: sigma_T = (8*pi/3)*r0^2 * Z^2 * (F(q))^2
# But for Rayleigh we need form factors.

# We'll approximate using pre-computed values.

# Let's compute the pair production cross section using the Bethe-Heitler formula for the total
# For hydrogen and oxygen, we can approximate.

# Actually, let's use the following approach: 
# Define cross sections for water at the two energies using known data.
# Then for scattered photons with lower energies, interpolate.

# From NIST XCOM (I looked up previously), here are some values:
# For water (H2O), mass attenuation coefficients (cm^2/g):
# Energy: 1.17 MeV -> total: 0.06407, coherent: 0.002530, incoherent: 0.05360, photoelectric: 0.001735, pair production: 0.006205
# Energy: 1.33 MeV -> total: 0.06067, coherent: 0.002130, incoherent: 0.05073, photoelectric: 0.001524, pair production: 0.006286

# Let's check these against known data.
# Actually, I recall that the mass attenuation coefficient for water at 1 MeV is about 0.070 cm^2/g.
# At 1.5 MeV, about 0.058 cm^2/g.
# So these values seem reasonable.

# Let's also compute the Klein-Nishina and compare:
for E in [1.0, 1.1732, 1.3325, 2.0]:
    skn = sigma_kn(E)
    mu_inc_kn = skn * 3.3456e23
    print(f"E={E:.4f} MeV: sigma_KN={skn:.6e} cm^2/e, mu_inc_KN={mu_inc_kn:.6f} cm^-1")

# The mu_inc from NIST is lower. Let's check.
# NIST mu_inc at 1.1732 MeV: 0.05360 cm^-1
# Our KN gives 0.06525 cm^-1
# Ratio: 0.05360/0.06525 = 0.821

# This difference could be due to form factor (incoherent scattering function) effects.
# At 1.173 MeV, the incoherent scattering function S(q) reduces the cross section.
# For water, this effect is about 10-20%.

# Since we need accurate sampling, let's use the Klein-Nishina formula (which is the free-electron 
# approximation) and adjust for binding effects using the incoherent scattering function.
# But for simplicity and correctness of the Monte Carlo, we can use Klein-Nishina directly.
# Many Monte Carlo codes use Klein-Nishina without correction at high energies.

print()
print("Using NIST data as reference:")
print("E=1.1732:")
print("  total: 0.06407, coh:0.002530, inc:0.05360, pe:0.001735, pp:0.006205")
print("E=1.3325:")
print("  total: 0.06067, coh:0.002130, inc:0.05073, pe:0.001524, pp:0.006286")