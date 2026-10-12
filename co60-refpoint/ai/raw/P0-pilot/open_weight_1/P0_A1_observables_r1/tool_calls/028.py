import pickle
import numpy as np
import math

with open('mc_results.pkl', 'rb') as f:
    data = pickle.load(f)

scoring = data['scoring']
elec_scoring = data['elec_scoring']
N = data['N']

# ===== CONSTANTS =====
cyl_volume = np.pi * 1.0**2 * 0.2  # 0.6283 cm^3
Phi0_per_src = np.pi * 0.2 / (121 * cyl_volume)  # 0.008264 cm^-2
print(f"N = {N}")
print(f"Phi0 per source photon = {Phi0_per_src:.6f}")

# ===== LAYER 1: PHOTON FLUENCE =====
prim_fluence_per_phi0 = scoring['ph_prim_fluence']/(cyl_volume*N)/Phi0_per_src
scat_fluence_per_phi0 = scoring['ph_scat_fluence']/(cyl_volume*N)/Phi0_per_src
total_fluence_per_phi0 = prim_fluence_per_phi0 + scat_fluence_per_phi0

print(f"\n=== LAYER 1 ===")
print(f"L1_primary_fluence_per_phi0 = {prim_fluence_per_phi0:.4f}")
print(f"L1_total_fluence_per_phi0 = {total_fluence_per_phi0:.4f}")
print(f"L1_scattered_fraction_fluence = {scat_fluence_per_phi0/total_fluence_per_phi0:.4f}")

# Energy fluence
photon_energy_edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

# Primary spectrum: all at 2 MeV (in the last bin... actually 2.0 > 1.4, so it's above the grid)
# The primary photons have energy 2.0 MeV, which is above the photon energy grid (max bin edge 1.4).
# So all primary fluence is in "the bin above 1.4" - but the grid only goes to 1.4.
# The problem says "photon_fluence_spectrum_primary" on the photon_energy grid (28 bins).
# Since primary photons are at 2.0 MeV > 1.4 MeV, there are no bins that contain them
# except possibly the last bin [1.35, 1.4]. But 2.0 > 1.4!
# So the primary spectrum is all zeros on this grid (the primary photons are at 2 MeV > 1.4).
# Hmm, this seems intentional: "both lines" in the definition of primary fluence, 
# suggesting the primary photons form a line at 2 MeV which is outside the plotted spectrum.

# Scattered spectrum
scat_spec = scoring['ph_scat_spec'] / (cyl_volume*N) / Phi0_per_src
prim_spec = scoring['ph_prim_spec'] / (cyl_volume*N) / Phi0_per_src

print(f"\nPrimary spectrum (28 bins):")
print(np.round(prim_spec, 6))
print(f"Sum: {np.sum(prim_spec):.4f}")
print(f"Scattered spectrum (28 bins):")
print(np.round(scat_spec, 6))
print(f"Sum: {np.sum(scat_spec):.4f}")

# Angular distributions
prim_ang = scoring['ph_prim_ang'] / np.sum(scoring['ph_prim_ang']) if np.sum(scoring['ph_prim_ang'])>0 else np.zeros(10)
scat_ang = scoring['ph_scat_ang'] / np.sum(scoring['ph_scat_ang']) if np.sum(scoring['ph_scat_ang'])>0 else np.zeros(10)
print(f"\nPrimary angular (should be all in cos>0.8):")
print(np.round(prim_ang, 6))
print(f"Scattered angular:")
print(np.round(scat_ang, 6))
print(f"Sum scattered: {np.sum(scat_ang):.4f}")

# Mean photon energy (fluence-weighted)
# For scattered: sum(E_center * fluence)/sum(fluence)
# Primary all at 2 MeV
scat_mean_E = 0
scat_total_L = np.sum(scoring['ph_scat_spec'])
for i in range(28):
    E_center = (photon_energy_edges[i] + photon_energy_edges[i+1])/2
    scat_mean_E += E_center * scoring['ph_scat_spec'][i]
scat_mean_E /= scat_total_L

# Include energies above 1.4? The scattered spectrum grid ends at 1.4.
# Scattered photons can have energy up to 2.0 (from small-angle Compton or Rayleigh).
# The track length for scattered photons above 1.4 goes into... nothing? 
# In my scoring, idx = searchsorted(photon_energy_edges, E, 'right') - 1
# For E=1.8, searchsorted gives idx_out = 28, so no bin gets it.
# Wait, let me check: photon_energy_edges has 29 elements. For E=1.8 (between 1.4 and above),
# searchsorted(edges, 1.8, 'right') = 28 (after the last edge). idx = 27, which IS a valid bin.
# Let me check: bins are edges[0]-edges[1], ..., edges[27]-edges[28].
# searchsorted returns insertion point. For E=1.8, which is > 1.4 (edges[28]):
# searchsorted(edges, 1.8, 'right') = 29 (insert after the last element).
# idx = 28, which is NOT < 28. So no scoring for E > 1.4!
# I need to handle this. The scattered photons with E > 1.4 should be counted.
# But the spectrum grid only goes to 1.4. So these photons fall outside the bins.
# The problem says the bins go to 1.4, so scattered photons above 1.4 are not in any bin.
# But their track length should still be in the total scattered fluence.

# Wait, but do we have scattered photons above 1.4? For first-scatter Compton at 2 MeV:
# E' = 2/(1+3.914*(1-cosθ)). For E' > 1.4: 1+3.914*(1-cosθ) < 1.4286 → 1-cosθ < 0.1095 → cosθ > 0.890.
# So photons scattered with angle < 27° have E' > 1.4. That's a significant fraction!
# In my MC, these photons contribute to ph_scat_fluence but NOT to ph_scat_spec.
# This means my scattered spectrum sum won't match the scattered fluence.
# I need to fix this for the "mean photon energy" but the spectrum bins are fixed by the grid.

# Actually, for the spectrum: the problem defines bins up to 1.4. Photons with E in (1.4, 2.0]
# are not in the 28 bins. Looking at the problem statement: "photon_energy (28 bins): 0, 0.05, ..., 1.4"
# So the spectrum should sum to the total fluence in those bins.
# But the photons above 1.4 are NOT counted. This is a known issue with the binning.

# For my calculation, I'll clamp: scattered photons with E > 1.4 are not in any bin.
# But their track length IS in the total scattered fluence.
# For the mean energy, I need to include them. Let me compute it properly.

# Let me recount: how much scattered track length is above 1.4?
# I need to modify the scoring. Actually, let me compute from the analytic/single-scatter.

# For now, let me estimate: from the single-scatter calculation, about 15-20% of scattered photons 
# have E > 1.4 MeV. With multiple scatter, this fraction is lower, maybe 10%.

# Let me use a simpler approach: compute mean scattered photon energy from the spectrum
# plus the tail above 1.4 compensated.

print(f"\nScattered track in bins: {scat_total_L:.2f}")
print(f"Total scattered track: {scoring['ph_scat_fluence']:.2f}")
print(f"Fraction in bins: {scat_total_L/scoring['ph_scat_fluence']:.4f}")
print(f"Mean scattered photon energy (in bins): {scat_mean_E:.3f} MeV")

# L1_mean_photon_energy: fluence-weighted mean of ALL photons
# = (sum_prim E*L + sum_scat E*L) / (sum_prim L + sum_scat L)
# For primary: all at E=2.0
# For scatter: need proper mean including photons above 1.4
# Let me adjust the scattered mean to account for the tail:
# From the spectrum shape, I can estimate the tail contribution.
# Actually, let me use the single-scatter result where I computed all energies.
# From the earlier single-scatter MC: mean scatter energy was 1.09 MeV.
# With multiple scatter, it's lower. Let me estimate ~0.75 MeV.

# Let me proceed and refine this. First the other quantities.

# ===== LAYER 2 =====
print(f"\n=== LAYER 2 ===")
cyl_int = scoring['cyl_int']
cyl_int_prim = scoring['cyl_int_prim']
cyl_int_scat = scoring['cyl_int_scat']
cyl_int_ET = scoring['cyl_int_energy_transfer']
total_ints = np.sum(cyl_int)
print(f"Total cylinder interactions: {total_ints}")
print(f"  Compton: {cyl_int[0]}, PE: {cyl_int[1]}, Pair: {cyl_int[2]}, Rayleigh: {cyl_int[3]}")
print(f"  Primary: {cyl_int_prim}")
print(f"  Scattered: {cyl_int_scat}")
print(f"  ET: {cyl_int_ET}")

# L2_interaction_fractions
print(f"L2_interaction_fractions:")
print(f"  Compton: {cyl_int[0]/total_ints:.4f}")
print(f"  Photoelectric: {cyl_int[1]/total_ints:.4f}")
print(f"  Pair: {cyl_int[2]/total_ints:.4f}")
print(f"  Rayleigh: {cyl_int[3]/total_ints:.4f}")

# Energy transfer fractions
total_ET = np.sum(cyl_int_ET)
print(f"L2_energy_transfer_fractions:")
print(f"  Compton: {cyl_int_ET[0]/total_ET:.4f}")
print(f"  Photoelectric: {cyl_int_ET[1]/total_ET:.4f}")
print(f"  Pair: {cyl_int_ET[2]/total_ET:.4f}")
print(f"  Rayleigh: {cyl_int_ET[3]/total_ET:.4f}")

# Primary interaction fractions
tot_prim = np.sum(cyl_int_prim)
print(f"L2_interaction_fractions_primary (total={tot_prim}):")
print(f"  Compton: {cyl_int_prim[0]/tot_prim:.4f}")
print(f"  Photoelectric: {cyl_int_prim[1]/tot_prim:.4f}")
print(f"  Pair: {cyl_int_prim[2]/tot_prim:.4f}")
print(f"  Rayleigh: {cyl_int_prim[3]/tot_prim:.4f}")

# Scattered interaction fractions
tot_scat = np.sum(cyl_int_scat)
print(f"L2_interaction_fractions_scattered (total={tot_scat}):")
print(f"  Compton: {cyl_int_scat[0]/tot_scat:.4f}")
print(f"  Photoelectric: {cyl_int_scat[1]/tot_scat:.4f}")
print(f"  Pair: {cyl_int_scat[2]/tot_scat:.4f}")
print(f"  Rayleigh: {cyl_int_scat[3]/tot_scat:.4f}")

# L2_scattered_share_of_interactions
print(f"L2_scattered_share_of_interactions: {tot_scat/total_ints:.4f}")

# ===== LAYER 3 =====
print(f"\n=== LAYER 3 ===")
elec_cyl = scoring['electrons_cylinder']
print(f"Electrons created in cylinder: {len(elec_cyl)}")

# L3_initial_spectrum
electron_energy_edges = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.4])
l3_hist = np.zeros(25)
l3_hist_prim = np.zeros(25)
l3_cos = []
for (T, cos_e, parent_prim) in elec_cyl:
    idx = np.searchsorted(electron_energy_edges, T, side='right') - 1
    if 0 <= idx < 25:
        l3_hist[idx] += 1
        if parent_prim:
            l3_hist_prim[idx] += 1
    l3_cos.append(cos_e)

norm_l3 = np.sum(l3_hist)
norm_l3_prim = np.sum(l3_hist_prim)
print(f"L3_initial_spectrum (fraction):")
print(np.round(l3_hist/norm_l3, 5) if norm_l3 > 0 else "No electrons")
print(f"L3_initial_spectrum_parent_primary:")
print(np.round(l3_hist_prim/norm_l3_prim, 5) if norm_l3_prim > 0 else "No primary electrons")

# Mean initial energy
mean_init_E = np.mean([T for (T,_,_) in elec_cyl]) if len(elec_cyl)>0 else 0
mean_init_E_prim = np.mean([T for (T,_,p) in elec_cyl if p]) if norm_l3_prim>0 else 0
print(f"L3_mean_initial_energy: {mean_init_E:.4f} MeV")
print(f"L3_mean_initial_energy_parent_primary: {mean_init_E_prim:.4f} MeV")

# L3_initial_direction
l3_cos_hist = np.histogram(l3_cos, bins=ct_edges)[0] if len(l3_cos)>0 else np.zeros(10)
l3_cos_hist = l3_cos_hist/np.sum(l3_cos_hist) if np.sum(l3_cos_hist)>0 else np.zeros(10)
print(f"L3_initial_direction:")
print(np.round(l3_cos_hist, 6))

# ===== LAYER 4 =====
print(f"\n=== LAYER 4 ===")
print(f"Electron fluence track: {elec_scoring['e_fluence']:.3f} cm")
print(f"Electron dose: {elec_scoring['e_dose']:.3f} MeV")

e_fluence_per_phi0 = elec_scoring['e_fluence']/(cyl_volume*N)/Phi0_per_src
print(f"L4_electron_fluence_per_phi0 = {e_fluence_per_phi0:.4f}")

e_spec = elec_scoring['e_spec']/(cyl_volume*N)/Phi0_per_src
print(f"L4_electron_fluence_spectrum:")
print(np.round(e_spec, 6))

# Mean electron energy
e_fluence_total = np.sum(elec_scoring['e_spec'])
mean_e_E = 0
for i in range(25):
    E_center = (electron_energy_edges[i]+electron_energy_edges[i+1])/2
    mean_e_E += E_center * elec_scoring['e_spec'][i]
mean_e_E /= e_fluence_total
print(f"L4_mean_electron_energy: {mean_e_E:.4f} MeV")

e_ang = elec_scoring['e_ang']/np.sum(elec_scoring['e_ang'])
print(f"L4_angular:")
print(np.round(e_ang, 6))
print(f"Sum: {np.sum(elec_scoring['e_ang']):.4f}")