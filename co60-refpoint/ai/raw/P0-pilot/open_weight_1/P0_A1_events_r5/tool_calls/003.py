import numpy as np

# Let me create a realistic model for photon interactions in water
# I'll use tabulated cross sections and Monte Carlo

# First, let me get approximate cross section data for water at various energies
# These are approximate values from NIST XCOM

# Energy (MeV), mu_rayleigh, mu_compton, mu_photo, mu_pair (cm^-1 for density 1.0)
# I'll create a set of data points

energy_data = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
                        0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,
                        1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])

# Rayleigh (coherent) - cm^-1
mu_R = np.array([0.00817, 0.00364, 0.00205, 0.00091, 0.00051, 0.00033, 0.00023, 
                 0.00013, 0.000082, 0.000036, 0.000020, 0.000009, 0.000005, 
                 0.000003, 0.000002, 0.0000015, 0.0000012,
                 0.0000008, 0.0000006, 0.0000004, 0.0000003, 0.0000002, 0.0000001,
                 0.0000001, 0.0000001, 0.0000001, 0.0000001])

# Actually these Rayleigh values seem too low at higher energies. Let me use more accurate data.
# Wait, at 2.0 MeV I computed mu_R = 0.000119 which is much higher than 4e-7
# Let me recalculate properly.

# Actually, the issue is that Rayleigh is coherent scattering and at high energies
# it's very forward-peaked. The mass attenuation coefficient for Rayleigh
# drops with energy. Let me check.

# From NIST XCOM for water at 2 MeV:
# Coherent (Rayleigh) scattering: μ/ρ = 0.000119 cm^2/g
# So μ = 0.000119 cm^-1 for density 1.0 g/cm^3

# The form factor suppresses Rayleigh at high energies but it's not zero.
# Let me use more accurate data.

# Let me use the NIST data more carefully. I'll estimate from known values.

# For water, let me use these approximate mass attenuation coefficients (cm^2/g)
# from NIST: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html

# Energy(MeV)  μ/ρ_total   μ/ρ_coh    μ/ρ_incoh   μ/ρ_photo   μ/ρ_pair
# I'll approximate based on the NIST data

# Let me use actual NIST data points for water

energies_nist = np.array([0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01,
                          0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
                          0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,
                          1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0])

# These are approximate values from NIST for water
# μ/ρ in cm^2/g for total, coherent, incoherent, photoelectric, pair production (nuclear + electron)

# Let me define approximate values based on NIST data
# I'll create arrays for the various components

# I'll use a simpler approach: use the known total and partial coefficients
# at the energies where I have them, and interpolate

# At 2.0 MeV (our source energy):
mu_2 = {'total': 0.04936, 'coherent': 0.000119, 'incoherent': 0.04877, 
        'photo': 0.000383, 'pair': 0.000089}

# At 1.0 MeV:
mu_1 = {'total': 0.07067, 'coherent': 0.000247, 'incoherent': 0.06958, 
        'photo': 0.000798, 'pair': 0.000049}

# At 0.5 MeV:
mu_05 = {'total': 0.09696, 'coherent': 0.00091, 'incoherent': 0.09477, 
         'photo': 0.00127, 'pair': 0.000013}

# At 0.2 MeV:
mu_02 = {'total': 0.1368, 'coherent': 0.00512, 'incoherent': 0.1277, 
         'photo': 0.00397, 'pair': 0.0}

# At 0.1 MeV:
mu_01 = {'total': 0.1707, 'coherent': 0.0195, 'incoherent': 0.1470, 
         'photo': 0.00422, 'pair': 0.0}

# At 0.05 MeV:
mu_005 = {'total': 0.2145, 'coherent': 0.0709, 'incoherent': 0.1379, 
          'photo': 0.00574, 'pair': 0.0}

# At 0.02 MeV:
mu_002 = {'total': 0.7987, 'coherent': 0.378, 'incoherent': 0.381, 
          'photo': 0.0397, 'pair': 0.0}

# At 0.01 MeV:
mu_001 = {'total': 5.329, 'coherent': 1.114, 'incoherent': 2.941, 
          'photo': 1.274, 'pair': 0.0}

print("Cross section data check:")
for name, data in [("2 MeV", mu_2), ("1 MeV", mu_1), ("0.5 MeV", mu_05), 
                     ("0.2 MeV", mu_02), ("0.1 MeV", mu_01), ("0.05 MeV", mu_005)]:
    total = data['coherent'] + data['incoherent'] + data['photo'] + data['pair']
    print(f"{name}: total={data['total']:.5f}, sum={total:.5f}, "
          f"C={data['incoherent']/data['total']*100:.1f}%, "
          f"PE={data['photo']/data['total']*100:.2f}%, "
          f"R={data['coherent']/data['total']*100:.2f}%, "
          f"PP={data['pair']/data['total']*100:.2f}%")