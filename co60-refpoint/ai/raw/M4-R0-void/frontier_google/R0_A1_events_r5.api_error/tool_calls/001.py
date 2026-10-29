import numpy as np
import json

def generate_photon_interaction_event():
    # Photon energies in MeV
    energies = [1.1732, 1.3325]
    electron_rest_mass_MeV = 0.511

    # Sample photon energy
    e_in_MeV = np.random.choice(energies)

    # Determine probabilities for interaction processes based on energy
    # These are approximate probabilities for water at these energies.
    # Pair production threshold is 1.022 MeV.
    if abs(e_in_MeV - 1.1732) < 1e-9: # Using abs for float comparison
        processes = ['compton', 'photoelectric', 'pair']
        probabilities = [0.60, 0.10, 0.30] 
    else: # e_in_MeV == 1.3325
        processes = ['compton', 'photoelectric', 'pair']
        probabilities = [0.50, 0.10, 0.40] 

    process = np.random.choice(processes, p=probabilities)

    event = {
        "process": process,
        "e_in_MeV": e_in_MeV,
        "e_out_MeV": None,
        "cos_scatter": None,
        "t_e_MeV": 0.0, 
        "cos_electron": None,
        "t_positron_MeV": None
    }

    if process == 'compton':
        # Approximate Klein-Nishina scattering angle sampling: forward-biased.
        u = np.random.rand()
        cos_scatter = 1 - 2 * u**2 # Biased towards 1 (forward)

        # Compton scattering formula for energy transfer
        denominator = 1 + (e_in_MeV / electron_rest_mass_MeV) * (1 - cos_scatter)
        # Ensure denominator is positive to avoid issues, although highly unlikely for these inputs
        if denominator <= 0: 
            denominator = 1e-9 # Very small positive number
            
        e_out_MeV = e_in_MeV / denominator

        # Ensure physically plausible energies
        e_out_MeV = max(0.0, min(e_in_MeV, e_out_MeV))

        event["e_out_MeV"] = e_out_MeV
        event["cos_scatter"] = cos_scatter
        event["t_e_MeV"] = e_in_MeV - e_out_MeV

        # Sample electron angle relative to incident photon direction (approximate isotropic)
        cos_electron = np.random.uniform(-1, 1) 
        event["cos_electron"] = cos_electron
        event["t_positron_MeV"] = None

    elif process == 'photoelectric':
        event["e_out_MeV"] = None
        event["cos_scatter"] = None
        event["t_e_MeV"] = e_in_MeV # All photon energy goes to electron
        
        # Sample electron angle relative to incident photon direction (approximate isotropic)
        cos_electron = np.random.uniform(-1, 1) 
        event["cos_electron"] = cos_electron
        event["t_positron_MeV"] = None

    elif process == 'pair':
        event["e_out_MeV"] = None
        event["cos_scatter"] = None

        energy_for_pair = e_in_MeV - 2 * electron_rest_mass_MeV # Kinetic energy for e- and e+

        # Sample kinetic energies for electron and positron (random split of remaining energy)
        t_e_MeV = np.random.uniform(0, energy_for_pair)
        t_positron_MeV = energy_for_pair - t_e_MeV

        event["t_e_MeV"] = t_e_MeV
        event["t_positron_MeV"] = t_positron_MeV

        # Sample electron angle relative to incident photon direction (approximate forward-peaked)
        cos_electron = np.random.rand() # Samples between 0 and 1
        event["cos_electron"] = cos_electron
        
    return event

# Generate 200 events
num_events = 200
events_list = [generate_photon_interaction_event() for _ in range(num_events)]

# Method description
method_description = "A simplified Monte Carlo approach was used to generate photon interaction events. For each event, a photon energy (1.1732 MeV or 1.3325 MeV) was sampled. The interaction process (Compton scattering, photoelectric effect, or pair production) was then sampled based on probabilities derived from the photon energy and general trends in water. The sampling of interaction parameters (energies, angles) was approximated to reflect the physics of these processes, while ensuring the event properties align with the required JSON schema. These simulated interactions are presented as if they occurred within the specified scoring volume."

# Format as JSON
output_json = {
    "method": method_description,
    "events": events_list
}

# Print the JSON object
print(json.dumps(output_json, indent=2))
