
import numpy as np
import json

np.random.seed(42) # for reproducibility

# Constants
E_IN_MEV = 2.0
MC2 = 0.511 # Electron rest mass energy in MeV
PAIR_THRESHOLD = 2 * MC2 # 1.022 MeV

# Probabilities of interaction processes at 2.0 MeV in water (approximate)
# These are rough estimates. Actual values require cross-section data.
PROCESS_PROBS = {
    "compton": 0.85,
    "photoelectric": 0.08,
    "pair": 0.06,
    "rayleigh": 0.01
}
PROCESSES = list(PROCESS_PROBS.keys())
PROBABILITIES = list(PROCESS_PROBS.values())

n_events = 200
events = []

# --- Sampling Helper Functions ---

def sample_compton():
    e_in = E_IN_MEV
    mc2 = MC2

    # Sample cos_scatter (approximating Klein-Nishina, forward-peaked)
    # Using a triangular distribution peaked at 0.8 for forward scattering
    cos_scatter = np.random.triangular(-1.0, 1.0, 0.8)

    # Calculate outgoing photon energy
    denominator = 1 + (e_in / mc2) * (1 - cos_scatter)
    e_out = e_in / denominator if denominator != 0 else e_in

    # Kinetic energy of recoil electron
    t_e = e_in - e_out

    # Cosine of electron direction relative to incident photon direction
    # Compton electron direction is somewhat forward-peaked
    cos_electron = np.random.uniform(0.3, 1.0)

    return {
        "process": "compton",
        "e_in_MeV": e_in,
        "e_out_MeV": e_out,
        "cos_scatter": cos_scatter,
        "t_e_MeV": t_e,
        "cos_electron": cos_electron,
        "t_positron_MeV": None
    }

def sample_photoelectric():
    e_in = E_IN_MEV
    # Photon is absorbed
    e_out = None
    cos_scatter = None

    # Kinetic energy of ejected electron (photoelectron)
    # Assuming binding energy of water atoms is negligible compared to 2 MeV
    t_e = e_in

    # Cosine of electron direction relative to incident photon direction
    # Photoelectrons tend to be forward-peaked
    cos_electron = np.random.uniform(0.5, 1.0)

    return {
        "process": "photoelectric",
        "e_in_MeV": e_in,
        "e_out_MeV": e_out,
        "cos_scatter": cos_scatter,
        "t_e_MeV": t_e,
        "cos_electron": cos_electron,
        "t_positron_MeV": None
    }

def sample_pair_production():
    e_in = E_IN_MEV
    mc2 = MC2
    # Photon is absorbed
    e_out = None
    cos_scatter = None

    # Kinetic energy available for electron and positron
    K_total = e_in - PAIR_THRESHOLD

    # Sample t_e uniformly from 0 to K_total.
    # This is a significant simplification of the Bethe-Heitler spectrum.
    t_e = np.random.uniform(0, K_total)
    t_positron = K_total - t_e

    # Cosine of electron direction relative to incident photon direction
    # Pair production angles are forward-peaked
    cos_electron = np.random.uniform(0.7, 1.0)

    return {
        "process": "pair",
        "e_in_MeV": e_in,
        "e_out_MeV": e_out,
        "cos_scatter": cos_scatter,
        "t_e_MeV": t_e,
        "cos_electron": cos_electron,
        "t_positron_MeV": t_positron
    }

def sample_rayleigh():
    e_in = E_IN_MEV
    # Elastic scattering, photon energy is conserved
    e_out = e_in

    # Cosine of scattering angle
    # Rayleigh scattering is highly forward-peaked.
    # Using Beta distribution peaked near 1.
    cos_scatter = np.random.beta(a=10, b=1)

    # No electron set in motion
    t_e = 0.0
    cos_electron = None

    return {
        "process": "rayleigh",
        "e_in_MeV": e_in,
        "e_out_MeV": e_out,
        "cos_scatter": cos_scatter,
        "t_e_MeV": t_e,
        "cos_electron": cos_electron,
        "t_positron_MeV": None
    }

# --- Generate Events ---
while len(events) < n_events:
    chosen_process = np.random.choice(PROCESSES, p=PROBABILITIES)

    event = None
    if chosen_process == "compton":
        event = sample_compton()
    elif chosen_process == "photoelectric":
        event = sample_photoelectric()
    elif chosen_process == "pair":
        # Check if energy is sufficient for pair production
        if E_IN_MEV > PAIR_THRESHOLD:
            event = sample_pair_production()
        else:
            continue # Skip this iteration and try another process
    elif chosen_process == "rayleigh":
        event = sample_rayleigh()

    if event: # Ensure event was successfully generated
        events.append(event)

# Final method description
method_description = (
    "A hypothetical Monte Carlo simulation approach is described for generating photon interaction events within a scoring volume. "
    "Photons are assumed to have an initial energy of 2.0 MeV upon interaction. "
    "The interaction process (Compton scattering, photoelectric effect, pair production, or Rayleigh scattering) "
    "is sampled based on approximate relative interaction probabilities for water at this energy. "
    "For each interaction, outgoing photon energies, scattering angles, and recoil particle kinetic energies and angles "
    "are sampled using simplified analytical distributions that mimic physical interactions. "
    "This method generates a plausible sample of interaction events but does not simulate full photon transport "
    "from the source or accurately represent the energy and angular spectra of scattered photons, as these "
    "require complex transport physics and cross-section data not available in this environment."
)

# Prepare the JSON output
output_data = {
    "method": method_description,
    "events": events
}

# Print the JSON. Ensure it's properly escaped for JSON.
print(json.dumps(output_data, indent=2))
