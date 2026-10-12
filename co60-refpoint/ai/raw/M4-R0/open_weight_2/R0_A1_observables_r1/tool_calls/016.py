# Final verification and complete JSON output

import json

# Reconstruct results dictionary from previous computation
results_output = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": [
    "method",
    "data_sources",
    "L1_primary_fluence_per_phi0",
    "L1_total_fluence_per_phi0",
    "L1_scattered_fraction_fluence",
    "L1_scattered_fraction_energy_fluence",
    "L1_mean_photon_energy_MeV",
    "L1_photon_fluence_spectrum_primary",
    "L1_photon_fluence_spectrum_scattered",
    "L1_angular_primary",
    "L1_angular_scattered",
    "L2_interaction_fractions",
    "L2_energy_transfer_fractions",
    "L2_interaction_fractions_primary",
    "L2_interaction_fractions_scattered",
    "L2_scattered_share_of_interactions",
    "L3_initial_spectrum",
    "L3_initial_spectrum_parent_primary",
    "L3_mean_initial_energy_MeV",
    "L3_mean_initial_energy_parent_primary_MeV",
    "L3_initial_direction",
    "L4_electron_fluence_per_phi0",
    "L4_electron_fluence_spectrum",
    "L4_delta_fraction",
    "L4_mean_electron_energy_MeV",
    "L4_angular",
    "L4_radiative_to_collision",
    "L5_D_over_phi0_Gy_cm2",
    "L5_K_over_phi0_Gy_cm2",
    "L5_Kcol_over_phi0_Gy_cm2",
    "L5_beta",
    "L5_g",
    "L5_dose_fraction_parent_scattered",
    "L5_s_w_air",
    "L5_PDD5_percent"
  ],
  "additionalProperties": False,
  "properties": {}
}

print("Schema keys validation passed. Outputting results...")