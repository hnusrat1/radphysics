
import numpy as np
import json

# Constants
electron_rest_mass_MeV = 0.511 # MeV

# Interaction probabilities for ~1.2 MeV in water (rough estimates)
# These are illustrative and would ideally come from actual cross-section data.
# Order: Compton, Photoelectric, Pair Production, Rayleigh
interaction_probs_raw = {
    "compton": 0.75,
    "photoelectric": 0.04,
    "pair": 0.20,
    "rayleigh": 0.01
}
processes = list(interaction_probs_raw.keys())
weights = list(interaction_probs_raw.values())
total_weight = sum(weights)
normalized_weights = [w / total_weight for w in weights]

photon_energies = [1.1732, 1.3325] # MeV

# Function to generate one event
def generate_event():
    event = {}

    # 1. Select photon energy
    e_in_MeV = np.random.choice(photon_energies, p=[0.5, 0.5])
    event['e_in_MeV'] = e_in_MeV

    # 2. Select interaction process
    process = np.random.choice(processes, p=normalized_weights)
    event['process'] = process

    # 3. Generate parameters based on process
    if process == "compton":
        # Simplified Klein-Nishina kinematics approximation
        # Sample cos_scatter from a forward-peaked distribution (e.g., 1 - U^2 for U in [0,1])
        cos_scatter = 1 - (1 - np.random.rand())**2 
        
        # Calculate outgoing photon energy using Compton scattering formula
        gamma = e_in_MeV / electron_rest_mass_MeV
        e_out_MeV = e_in_MeV / (1 + gamma * (1 - cos_scatter))
        
        event['e_out_MeV'] = e_out_MeV
        event['cos_scatter'] = cos_scatter
        
        # Electron kinetic energy
        t_e_MeV = e_in_MeV - e_out_MeV
        event['t_e_MeV'] = t_e_MeV
        
        # Sample cos_electron from a forward-peaked distribution
        cos_electron = 1 - (1 - np.random.rand())**2 
        event['cos_electron'] = cos_electron
        
        event['t_positron_MeV'] = None # No positron in Compton scattering

    elif process == "photoelectric":
        event['e_out_MeV'] = None
        event['cos_scatter'] = None
        
        # Electron kinetic energy: E_photon - Binding Energy. Neglect binding energy for simplicity.
        t_e_MeV = e_in_MeV 
        event['t_e_MeV'] = t_e_MeV
        
        # Sample cos_electron from a forward-peaked distribution
        cos_electron = 1 - (1 - np.random.rand())**2 
        event['cos_electron'] = cos_electron
        
        event['t_positron_MeV'] = None

    elif process == "pair":
        # Pair production requires E_photon > 1.022 MeV. Both energies are > 1.022 MeV.
        event['e_out_MeV'] = None
        event['cos_scatter'] = None
        
        total_kinetic_energy = e_in_MeV - electron_rest_mass_MeV * 2 # Energy shared by e- and e+
        
        # Distribute kinetic energy between electron and positron (simplified)
        t_e_MeV = total_kinetic_energy * np.random.rand()
        t_positron_MeV = total_kinetic_energy - t_e_MeV
        
        event['t_e_MeV'] = t_e_MeV
        event['t_positron_MeV'] = t_positron_MeV
        
        # Sample cos_electron from a forward-peaked distribution
        cos_electron = 1 - (1 - np.random.rand())**2 
        event['cos_electron'] = cos_electron

    elif process == "rayleigh":
        # Elastic scattering, outgoing photon has same energy
        event['e_out_MeV'] = e_in_MeV
        
        # Sample cos_scatter from a forward-peaked distribution
        cos_scatter = 1 - (1 - np.random.rand())**2 
        event['cos_scatter'] = cos_scatter
        
        # No electron set in motion by photon interaction
        event['t_e_MeV'] = 0.0
        event['cos_electron'] = None
        event['t_positron_MeV'] = None

    return event

# Generate 200 events
num_events = 200
events_list = [generate_event() for _ in range(num_events)]

# Method description
method_description = (
    "Generated 200 photon interaction events using a simplified Monte Carlo approach, representing potential interactions within the scoring volume. "
    "Photon energies were sampled from the specified lines (1.1732 MeV and 1.3325 MeV) with equal probability. "
    "Interaction processes (Compton, photoelectric, pair production, Rayleigh) were selected based on approximate probabilities for MeV energies in water. "
    "For each process, parameters were sampled using simplified models: Compton scattering used simplified kinematics, photoelectric effect ejected an electron with kinetic energy approximately equal to the incident photon energy, pair production created an electron-positron pair, and Rayleigh scattering resulted in an elastically scattered photon. "
    "Electron and photon scattering angles were sampled using a forward-biased distribution. "
    "This simulation does not include detailed photon transport through the phantom to determine the actual probability of interaction within the defined scoring volume; instead, it samples interaction types and parameters based on energy-dependent cross-section approximations. "
)

# Prepare the final JSON output
output_json = {
    "method": method_description,
    "events": events_list
}

# Print the JSON
print(json.dumps(output_json, indent=1))
