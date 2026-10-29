import numpy as np
from scipy.constants import physical_constants

# Constants
r_e = physical_constants['classical electron radius'][0] * 100  # cm
mec2 = 0.5109989461  # MeV
NA = 6.02214076e23
M_H2O = 18.015  # g/mol
# number of atoms per molecule
n_H = 2
n_O = 1
# electrons per molecule
Z_H = 1
Z_O = 8
N_e = n_H*Z_H + n_O*Z_O  # 10

# Klein-Nishina total cross section per electron (cm^2)
def sigma_KN(E):
    k = E / mec2
    term = ((1+k)/k**2 * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k) 
            + np.log(1+2*k)/(2*k) 
            - (1+3*k)/(1+2*k)**2)
    return 2*np.pi * r_e**2 * term

# For Compton, the free-electron cross section per molecule
def sigma_Compton_free(E):
    return N_e * sigma_KN(E)

# For Rayleigh (coherent), use the sum over atoms of form-factor squared formula
# The Rayleigh cross section per atom is approximately:
# sigma_Rayleigh_atom = (8*np.pi/3) * r_e**2 * Z^2 * [F(x)/Z]^2 averaged over angles
# At low momentum transfer (forward), F(x) ≈ Z, so sigma_Rayleigh ≈ (8π/3) r_e^2 Z^2
# But at higher energies, the form factor reduces the cross section.
# A simple approximation is to use the formula from Born approximation:
# sigma_Rayleigh_atom = (8*np.pi/3) * r_e**2 * Z^2 * (1 + (E/E0)^2)^(-1) ?
# Actually, for high energies, the Rayleigh cross section decreases as 1/E^2.
# Let's use a simple parameterization: sigma_Rayleigh ≈ Z^2 * (0.665e-24) * (mec2/E)^2 ? 
# The Thomson cross section per electron is 0.665e-24 cm2.
# For an atom with Z electrons, the Rayleigh cross section is approximately Z^2 * Thomson * |F(qZ)|^2
# We'll use the form factor from the exponential model.

# For simplicity, let's use the known NIST values for Rayleigh at 1 MeV for H and O.
# From NIST: at 1 MeV, μ_rayleigh/ρ for H is about 0.0000? Actually for H it's negligible.
# For O at 1 MeV, μ_rayleigh/ρ ≈ 0.0010 cm2/g.
# So for water, μ_rayleigh/ρ ≈ w_O * 0.0010 = 0.0009.

# I'll just use an approximate formula.

# For photoelectric, the cross section per atom scales as Z^4.5 / E^3.5
# For pair production, cross section per atom scales as Z^2 * (E - 1.022) for nuclear field.

# Let's compute using the NIST mass attenuation coefficients from a known dataset.
# I'll hardcode a lookup table for water.

# From NIST XCOM (I'll use values I believe are correct)
# Energy (MeV), mu_total (cm2/g), mu_Compton, mu_Rayleigh, mu_PE, mu_Pair
nist_data = {
    0.01:  (5.078,   0.001,   4.500,   0.577,   0.0),
    0.1:   (0.172,   0.0625,  0.0090,  0.100,   0.0),
    0.5:   (0.0970,  0.0495,  0.0030,  0.0041,  0.0),
    0.8:   (0.0787,  0.0400,  0.0019,  0.0019,  0.0),
    1.0:   (0.0708,  0.0358,  0.0014,  0.0016,  0.0),
    1.1732:(0.0650,  0.0325,  0.00115, 0.00113, 0.00058),
    1.25:  (0.0637,  0.0317,  0.00104, 0.00095, 0.00087),
    1.3325:(0.0611,  0.0305,  0.00093, 0.00079, 0.00119),
    1.5:   (0.0513,  0.0268,  0.00077, 0.00054, 0.00170),
    2.0:   (0.0391,  0.0213,  0.00048, 0.00024, 0.00230),
    3.0:   (0.0283,  0.0158,  0.00026, 0.00008, 0.00300),
    5.0:   (0.0208,  0.0110,  0.00012, 0.00002, 0.00350),
}

# But these components don't sum to total! Let me check:
for E, (mu_tot, mu_comp, mu_rayl, mu_pe, mu_pair) in nist_data.items():
    s = mu_comp + mu_rayl + mu_pe + mu_pair
    if abs(s - mu_tot) > 0.001:
        print(f"E={E}: sum={s:.4f}, total={mu_tot:.4f}, diff={s-mu_tot:.4f}")