import numpy as np
from scipy.stats import beta

# Constants
me_energy = 0.511 # MeV

def generate_event():
    # 1. Sample Incident Photon Energy
    e_in = np.random.choice([1.1732, 1.3325], p=[0.5, 0.5])

    # 2. Sample Interaction Process based on NNDC/XCOM data for water at ~1.2 MeV
    # Probabilities for 1.3325 MeV in water: Compton 96.8%, Photo 2.5%, Pair 0.7%
    process_probs = {
        'compton': 0.968,
        'photoelectric': 0.025,
        'pair': 0.007
    }
    # Normalize probabilities in case of minor rounding issues
    total_prob = sum(process_probs.values())
    if not np.isclose(total_prob, 1.0):
        for p in process_probs:
            process_probs[p] /= total_prob

    process = np.random.choice(list(process_probs.keys()), p=list(process_probs.values()))

    # 3. Define event properties based on process
    event = {
        "process": process,
        "e_in_MeV": e_in,
        "e_out_MeV": None,
        "cos_scatter": None,
        "t_e_MeV": 0.0, # Default, will be updated
        "cos_electron": None,
        "t_positron_MeV": None
    }

    if process == 'compton':
        # Simplified Klein-Nishina scattering angle distribution (favors forward)
        # Sample theta from a distribution biased towards 0. beta(2,0.5) is suitable for forward bias.
        theta_sample = np.pi * beta.rvs(2, 0.5)
        cos_scatter_sample = np.cos(theta_sample)

        # Calculate outgoing photon energy using simplified Compton formula
        alpha = e_in / me_energy
        e_out_sample = e_in / (1 + alpha * (1 - cos_scatter_sample))

        event["e_out_MeV"] = e_out_sample
        event["cos_scatter"] = cos_scatter_sample
        event["t_e_MeV"] = e_in - e_out_sample

        # Sample electron direction cosine (forward-biased)
        # Use beta(2,1) for a distribution biased towards +1
        event["cos_electron"] = 2 * beta.rvs(2, 1) - 1

    elif process == 'photoelectric':
        event["e_out_MeV"] = None
        event["cos_scatter"] = None
        # Incident photon energy is converted to electron kinetic energy (neglecting binding energy)
        event["t_e_MeV"] = e_in
        # Electron emission angle for photoelectric effect is not simply isotropic.
        # For light elements, it has a forward bias. Using uniform as a basic placeholder.
        event["cos_electron"] = np.random.uniform(-1, 1) # Simplified isotropic emission

    elif process == 'pair':
        # Energy available for pair = e_in - 2*me_energy
        energy_for_pair = e_in - 2 * me_energy

        # Distribution of energy sharing between electron and positron.
        # Beta(2,2) implies roughly equal sharing with some variance.
        f_electron = beta.rvs(2, 2) # Fraction of energy for electron (0 to 1)
        event["t_e_MeV"] = energy_for_pair * f_electron
        event["t_positron_MeV"] = energy_for_pair * (1 - f_electron)

        # Angular distribution for pair production is forward-peaked.
        # Use beta(2,1) for forward-biased electron direction.
        event["cos_electron"] = 2 * beta.rvs(2, 1) - 1

    # Post-processing to ensure values are within physical bounds and handle potential numerical issues.
    if event["t_e_MeV"] is not None and event["t_e_MeV"] < 0:
        event["t_e_MeV"] = 0.0
    if event["t_positron_MeV"] is not None and event["t_positron_MeV"] < 0:
        event["t_positron_MeV"] = 0.0
    if event["e_out_MeV"] is not None:
        if event["e_out_MeV"] < 0:
            event["e_out_MeV"] = 0.0
        if event["e_out_MeV"] > event["e_in_MeV"]:
            event["e_out_MeV"] = event["e_in_MeV"]
    
    # Ensure cos_electron is set for non-Rayleigh processes (which are excluded here)
    # If cos_electron ended up as None for a process where it's required, set a default.
    # This branch should not be strictly necessary given current logic but good for robustness.
    if event["cos_electron"] is None and event["process"] != 'rayleigh': # Rayleigh is not generated
        event["cos_electron"] = np.random.uniform(-1, 1)


    return event

# Generate 200 events
all_events = [generate_event() for _ in range(200)]

# Construct the output JSON
output_json = {
    "method": "Generated 200 photon interaction events using simplified physics models in Python. Incident photon energies were sampled from 1.1732 MeV and 1.3325 MeV with equal probability. Interaction processes were chosen based on approximate cross-section data for water at these energies (primarily Compton scattering, with small fractions for photoelectric and pair production). For Compton scattering, outgoing photon energy and electron kinetic energy were calculated using a simplified Klein-Nishina approximation, with scattering angles biased towards forward directions. For photoelectric effect, incident photon energy was assigned as electron kinetic energy. For pair production, energy was split between electron and positron, with forward-biased angular distributions. These are approximations and not derived from a full Monte Carlo simulation.",
    "events": all_events
}

import json
print(json.dumps(output_json, indent=2))
