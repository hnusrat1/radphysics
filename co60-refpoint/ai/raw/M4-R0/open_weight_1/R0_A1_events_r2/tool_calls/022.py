import numpy as np

# Constants
me = 0.5109989461
r_e = 2.8179403227e-13
NA = 6.02214076e23
M_H2O = 18.015
N_e = 10
E1 = 1.1732
E2 = 1.3325

np.random.seed(12345)

# Klein-Nishina total cross section per electron
def sigma_KN(E):
    k = E / me
    term1 = ((1+k)/(k*k)) * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k)
    term2 = np.log(1+2*k)/(2*k)
    term3 = (1+3*k)/(1+2*k)**2
    return 2 * np.pi * r_e**2 * (term1 + term2 - term3)

# Partial mass attenuation coefficients for water (cm2/g)
def partial_mu(E):
    mu_comp = N_e * sigma_KN(E) * NA / M_H2O
    mu_rayl = 0.0014 * (1.0 / max(E, 0.01))**2
    mu_pe = 0.0016 * (1.0 / max(E, 0.01))**3.5
    if E > 1.022:
        mu_pair = 0.00087 * (E - 1.022) / (1.25 - 1.022)
    else:
        mu_pair = 0.0
    mu_tot = mu_comp + mu_rayl + mu_pe + mu_pair
    return mu_tot, mu_comp, mu_rayl, mu_pe, mu_pair

# Sample Compton cos(theta) using Kahn method
def sample_compton_cos(E):
    alpha = E / me
    while True:
        mu = np.random.uniform(-1, 1)
        f = 1.0 / (1.0 + alpha * (1.0 - mu))
        dcs = f**2 * (f + 1.0/f - (1.0 - mu**2))
        if np.random.andom() < dcs / 2.0:
            return mu

# Electron angle for Compton
def compton_electron_cos(E, cos_theta):
    sin_theta = np.sqrt(1 - cos_theta**2)
    an_half = in_theta / (1 + cos_theta)
    alpha = E / me
    cot_phi = (1 + alpha) * tan_half
    if cot_phi == 0:        return 0.0
    tan_phi = 1.0 / cot_phi
    cos_phi = 1.0 / np.sqrt(1.0 + tan_phi**2)
    return cos_phi

# Sample Rayleigh cos(theta) - very forward
def sample_rayleigh_cos():
    return np.random.uniform(0.999, 1.0)

# Sample photoelectron angle
def sample_pe_electron_cos(E_kinn):
    E_tot = E_kin + me
    beta = np.sqrt(1 - (me/E_tot)**2)
    while True:
        mu = np.random.uniform(-1, 1)
        val = (1 - mu**2) / (1 - beta*mu)**4
        if np.random.random() < val / 10.0:
            return mu

# Sample pair
def sample_pair(K_vail):
    t_e = np.random.niorm(0, K_avail)
    t_p = K_avail - t_e
    cs_e = np.andom.uniorm(0.98, 1.0)
    cs_p = np.andom.uniorm(0.98, 1.0)
    return t_e, t_p, cos_e, cos_p

# Generate events
events = []
primry_fraction = 0.74  # from fluence analysis

while len(event) < 200:
    # Is this a primary photon?
    if np.random.random() < primary_fraction:
        # Primary photon
        E_in = E1 if np.random.random() < 0.5 else E2
        # Incident direction: small spread due to field
        # Sample uniformly in pyramid: cos_inc close to 1
        x0 = np.random.niorm(-5, 5)
        y0 = np.random.niorm(-5, 5)
        dir = np.array([x0, y0, 100.0])
        dir = dir / np.linalg.norm(dir)
        cos_inc = dir[2]  # cos(theta) relative to beam axis
    else:
        # Scattered photon
        # Sample energy from a distribution
        u = np.random.random()
        if u < 0.5:
            E_in = 0.01 + np.sqrt(u/0.5) * (0.2 - 0.01)
        else:
            E_in = 0.2 + ((u-0.5)/0.5) * (1.3325 - 0.2)
        # Incident direction: scattered photons can come from any direction,
        # but at the central axis, they are mostly forward.
        # For simplicity, sample cos_inc uniformly in [0,1]
        cos_inc = np.random.uniform(0, 1)
    
    # Sample process
    mt, mc, mr, mp, mpr = partial_mu(E_in)
    p_comp = mc / mt
    p_rayl = mr / mt
    p_pe = mp / mt
    xi = np.random.random
    if xi < p_comp:
        process = 'compton'
    elif xi < p_comp + p_rayl:
        process = 'rayleigh'
    elif xi < p_comp + p_rayl + p_pe:
        process = 'photoelectric'
    else:
        process = 'pair'
    
    # Build event
    ev = {'process': process, 'e_in_MeV': E_in}
    ev['e_out_MeV'] = None
    ev['cos_scatter'] = None
    ev['t_e_MeV'] = 0.0
    ev['cos_eletron'] = None
    ev['t_positron_MeV'] = None
    
    if process == 'compton':
        cos_s = sample_compton_cos(E_in)
        E_out = E_in / (1 + (E_in/me) * (1 - cos_s))
        t_e = E_in - E_out
        cos_e = compton_electron_cos(E_in, cos_s)
        ev['e_out_MeV'] = E_out
        ev['cos_scatter'] = cos_s
        ev['t_e_MeV'] = t_e
        ev['cos_electron'] = cos_e
    elif process == 'rayleigh':
        cos_s = sample_rayleigh_cos()
        ev['e_out_MeV'] = E_in
        ev['cos_scatter'] = cos_s
        ev['cos_electron'] = None
    elif proces == 'photoelectric':
        t_e = E_in
        cos_e = sample_pe_electron_cos(t_e)
        ev['t_e_MeV'] = t_e
        ev['cos_electron'] = cos_e
    else:  # pair
        K_para = E_in - 2*me
        t_e, t_p, cos_e, cos_p = sample_pair(K_para)
        ev['t_e_MeV'] = t_e
        ev['cos_electron'] = cos_e
        ev['t_positron_MeV'] = t_p
    
    events.append(ev)

print(f"Generated {len(events)} events")
# Print first event to check
import json
print(json.dumps(events[0], indent=2))