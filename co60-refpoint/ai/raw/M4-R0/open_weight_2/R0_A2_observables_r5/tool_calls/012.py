import numpy as np
from scipy.special import eval_legendre

# Reload data
with open('/tmp/data/xcom_water.txt', 'r') as f:
    xcom_lines = f.readlines()
with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

xcom_data = [np.array([float(p) for p in line.split()]) 
             for line in xcom_lines[4:] if len(line.split()) >= 7]
xcom_array = np.array(xcom_data)

estar_data = [np.array([float(p) for p in line.split()]) 
              for line in estar_lines[4:] if len(line.split()) >= 7]
estar_array = np.array(estar_data)

def interpolate_xlog(E, col_idx):
    energies = xcom_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    t = (np.log10(E) - log_E_low) / (log_E_high - log_E_low)
    val_low, val_high = xcom_array[idx-1, col_idx], xcom_array[idx, col_idx]
    if val_low > 0 and val_high > 0:
        return val_low * (val_high/val_low)**t
    elif val_high > 0:
        return val_high
    return 0.0

def interpolate_estar_log(E):
    energies = estar_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies):
        return None
    log_E_low = np.log10(energies[idx-1])
    log_E_high = np.log10(energies[idx])
    t = (np.log10(E) - log_E_low) / (log_E_high - log_E_low)
    result = {}
    for name, idx_col in [('collision',1), ('radiative',2), ('total',3)]:
        val_low, val_high = estar_array[idx-1, idx_col], estar_array[idx, idx_col]
        if val_low > 0 and val_high > 0:
            result[name] = val_low * (val_high/val_low)**t
        else:
            result[name] = val_high if val_high > 0 else 0.0
    return result

E_co1, E_co2 = 1.1732, 1.3325
rho = 1.0
depth = 5.0

# Cross sections
xs = {}
for E in [E_co1, E_co2]:
    xs[E] = {k: interpolate_xlog(E, i+1) for i, k in enumerate(['coherent','incoherent','photoelectric','pair_nuclear','pair_electron','total'])}

print("Layer 2: Photon Interactions")

# Interaction fractions by type (averaged over two lines)
frac_types_avg = {}
for key in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    if key == 'compton':
        key_xs = 'incoherent'
    elif key == 'photoelectric':
        key_xs = 'photoelectric'
    elif key == 'pair':
        key_xs = 'pair_nuclear' + xs[E_co2]['pair_electron']  # Sum pair contributions
        frac_types_avg[key] = 0.5*(xs[E_co1]['pair_nuclear']+xs[E_co1]['pair_electron'])/(xs[E_co1]['total']) \
                             + 0.5*(xs[E_co2]['pair_nuclear']+xs[E_co2]['pair_electron'])/(xs[E_co2]['total'])
        continue
    else:
        key_xs = 'coherent'
    
    frac_types_avg[key] = 0.5*xs[E_co1][key_xs]/xs[E_co1]['total'] + 0.5*xs[E_co2][key_xs]/xs[E_co2]['total']

# Better calculation for pair
frac_types_avg['pair'] = 0.5*(xs[E_co1]['pair_nuclear']+xs[E_co1]['pair_electron'])/(xs[E_co1]['total']) \
                        + 0.5*(xs[E_co2]['pair_nuclear']+xs[E_co2]['pair_electron'])/(xs[E_co2]['total'])
frac_types_avg['compton'] = 0.5*xs[E_co1]['incoherent']/xs[E_co1]['total'] + 0.5*xs[E_co2]['incoherent']/xs[E_co2]['total']
frac_types_avg['photoelectric'] = 0.5*xs[E_co1]['photoelectric']/xs[E_co1]['total'] + 0.5*xs[E_co2]['photoelectric']/xs[E_co2]['total']
frac_types_avg['rayleigh'] = 0.5*xs[E_co1]['coherent']/xs[E_co1]['total'] + 0.5*xs[E_co2]['coherent']/xs[E_co2]['total']

print(f"Interaction fractions (average):")
for k,v in frac_types_avg.items():
    print(f"  {k}: {v:.6f}")
print(f"  Sum: {sum(frac_types_avg.values()):.6f}")

# Energy transfer fractions
# For Compton: weighted by mean energy transfer <T>/E ~ 0.37 for Co-60
# For photoelectric: essentially full energy (~1.0)
# For pair: ~E - 1.022 MeV transferred (minus annihilation)
# For Rayleigh: 0

alpha_co1 = E_co1/0.511
alpha_co2 = E_co2/0.511

# Mean Compton energy transfer fraction
def mean_compton_frac(E):
    costheta = np.linspace(-1, 1, 1000)
    alpha = E/0.511
    sigma_ws = []
    T_over_E_ws = []
    for ct in costheta:
        theta = np.arccos(np.clip(ct, -1, 1))
        E_prime = E / (1 + alpha*(1-ct))
        T = E - E_prime
        r0sq = 0.066524e-24/(4*np.pi)
        factor = (1+ct**2)/2
        epe = E_prime/E
        sig = factor * epe**2 * r0sq
        sigma_ws.append(sig)
        T_over_E_ws.append(T/E)
    sigma_ws = np.array(sigma_ws)
    T_over_E_ws = np.array(T_over_E_ws)
    return np.sum(T_over_E_ws*sigma_ws)/np.sum(sigma_ws)

mean_T_over_E_co1 = mean_compton_frac(E_co1)
mean_T_over_E_co2 = mean_compton_frac(E_co2)
print(f"\nMean Compton energy transfer fractions:")
print(f"  E={E_co1}: {mean_T_over_E_co1:.4f}")
print(f"  E={E_co2}: {mean_T_over_E_co2:.4f}")

# Energy transfer fractions weighted
# Primary photons only first
energy_transfer_fractions = {}
for key in ['compton', 'photoelectric', 'pair', 'rayleigh']:
    if key == 'compton':
        avg_frac = 0.5*mean_T_over_E_co1 + 0.5*mean_T_over_E_co2
    elif key == 'photoelectric':
        avg_frac = 1.0  # Full energy transfer
    elif key == 'pair':
        # Average energy transfer = E - 1.022 (approx, ignoring recoil)
        avg_frac = lambda E: max(0, (E - 1.022)/E)
        avg_frac = 0.5*(E_co1-1.022)/E_co1 + 0.5*(E_co2-1.022)/E_co2
    else:  # rayleigh
        avg_frac = 0.0
    
    weight = frac_types_avg[key]
    energy_transfer_fractions[key] = weight * avg_frac

# Normalize
et_sum = sum(energy_transfer_fractions.values())
for key in energy_transfer_fractions:
    energy_transfer_fractions[key] /= et_sum

print(f"\nEnergy transfer fractions:")
for k,v in energy_transfer_fractions.items():
    print(f"  {k}: {v:.6f}")