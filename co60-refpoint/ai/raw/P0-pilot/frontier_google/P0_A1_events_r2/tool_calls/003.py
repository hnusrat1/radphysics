
import numpy as np
import json

# Constants
m_e = 0.511  # MeV (electron rest mass)
photon_energy = 2.0  # MeV (monoenergetic photons)
pair_production_threshold = 1.022  # MeV (for electron-positron pair)

# Interaction probabilities based on typical cross-sections for water at 2 MeV
# These are approximate and normalized.
interaction_probs = {
    "compton": 0.75,
    "pair": 0.20,
    "photoelectric": 0.03,
    "rayleigh": 0.02
}
processes = list(interaction_probs.keys())
probs = list(interaction_probs.values())

events = []
num_events = 200

for _ in range(num_events):
    event = {
        "process": None,
        "e_in_MeV": photon_energy,
        "e_out_MeV": None,
        "cos_scatter": None,
        "t_e_MeV": 0.0,
        "cos_electron": None,
        "t_positron_MeV": None
    }

    # Sample interaction process
    process = np.random.choice(processes, p=probs)
    event["process"] = process

    if process == "compton":
        # Compton scattering
        alpha = photon_energy / m_e
        
        # Simplified sampling for cos_theta from Klein-Nishina distribution.
        # This method uses uniform sampling for cos_theta and calculates other variables based on it.
        # Note: This is a simplification and does not accurately represent the Klein-Nishina angular distribution.
        cos_theta_val = np.random.uniform(-1.0, 1.0)
        
        # Calculate outgoing photon energy and electron kinetic energy using Compton kinematics
        # E' = E / (1 + alpha * (1 - cos(theta)))
        e_out_val = photon_energy / (1 + alpha * (1 - cos_theta_val))
        t_e_val = photon_energy - e_out_val
        
        event["e_out_MeV"] = e_out_val
        event["cos_scatter"] = cos_theta_val
        event["t_e_MeV"] = t_e_val
        
        # Simplified sampling for electron angle (cosine of angle with incident photon direction)
        event["cos_electron"] = np.random.uniform(-1.0, 1.0)

    elif process == "photoelectric":
        # Photoelectric effect
        event["e_out_MeV"] = None
        event["cos_scatter"] = None
        
        # Electron kinetic energy is E_in - Binding Energy. For water at 2 MeV, binding energies are small.
        # We approximate t_e_MeV ~ photon_energy.
        event["t_e_MeV"] = photon_energy 
        
        # Simplified sampling for electron angle
        event["cos_electron"] = np.random.uniform(-1.0, 1.0)

    elif process == "pair":
        # Pair production
        event["e_out_MeV"] = None
        event["cos_scatter"] = None
        
        kinetic_energy_total = photon_energy - pair_production_threshold
        if kinetic_energy_total > 0:
            # Sample energy split between electron and positron kinetic energies
            t_e_val = np.random.uniform(0, kinetic_energy_total)
            t_positron_val = kinetic_energy_total - t_e_val
            
            event["t_e_MeV"] = t_e_val
            event["t_positron_MeV"] = t_positron_val
            
            # Simplified sampling for electron angle
            event["cos_electron"] = np.random.uniform(-1.0, 1.0)
        else:
            # This case should not occur for 2.0 MeV photons, but as a safeguard:
            event["t_e_MeV"] = 0.0
            event["cos_electron"] = None # No particles created

    elif process == "rayleigh":
        # Rayleigh scattering (coherent scattering)
        event["e_out_MeV"] = photon_energy  # Energy is conserved
        
        # Rayleigh scattering angles are typically small. Sample cos_scatter biased towards 1.
        # Using uniform sampling within a limited range for simplification.
        event["cos_scatter"] = np.random.uniform(-0.95, 0.95)
        
        event["t_e_MeV"] = 0.0 # No electron set in motion
        event["cos_electron"] = None # No electron
        event["t_positron_MeV"] = None # No positron

    events.append(event)

# Prepare JSON output
output = {
    "method": "Simplified Monte Carlo simulation of photon interactions in water. Interaction types (Compton, Photoelectric, Pair Production, Rayleigh) are sampled based on approximate probabilities: Compton (75%), Pair Production (20%), Photoelectric Effect (3%), Rayleigh Scattering (2%). For each process, kinematic variables are generated using simplified models. Compton scattering uses uniform sampling for the photon scattering angle, with derived electron and outgoing photon energies following the kinematic relationship. Pair production incident photon energy minus threshold is split between electron and positron kinetic energies. Photoelectric effect yields an electron with kinetic energy approximately equal to the incident photon energy. Rayleigh scattering conserves photon energy. Angular distributions for secondary electrons are sampled uniformly. This method provides structurally correct output but does not represent a strictly unbiased physical sample due to approximations in sampling physics distributions and lack of full transport simulation.",
    "events": events
}

# Print the JSON object
print(json.dumps(output, indent=1))
