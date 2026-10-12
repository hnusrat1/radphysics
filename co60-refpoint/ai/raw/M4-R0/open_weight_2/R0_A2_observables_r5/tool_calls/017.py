import numpy as np

# Debug kerma calculation
with open('/tmp/data/xcom_water.txt', 'r') as f:
    xcom_lines = f.readlines()
with open('/tmp/data/estar_water.txt', 'r') as f:
    estar_lines = f.readlines()

xcom_data = []
for line in xcom_lines[4:]:
    if line.startswith('#'): continue
    parts = line.split()
    if len(parts) >= 7:
        try: xcom_data.append([float(p) for p in parts])
        except: pass
xcom_array = np.array(xcom_data)

estar_data = []
for line in estar_lines[4:]:
    if line.startswith('#'): continue
    parts = line.split()
    if len(parts) >= 7:
        try: estar_data.append([float(p) for p in parts])
        except: pass
estar_array = np.array(estar_data)

def interpolate_xlog(E, col_idx):
    energies = xcom_array[:, 0]
    idx = np.searchsorted(energies, E)
    if idx == 0 or idx >= len(energies): return None
    log_E_low = np.log10(max(energies[idx-1], 1e-20))
    log_E_high = np.log10(max(energies[idx], 1e-20))
    t = (np.log10(max(E, 1e-20)) - log_E_low) / (log_E_high - log_E_low)
    val_low, val_high = xcom_array[idx-1, col_idx], xcom_array[idx, col_idx]
    if val_low > 0 and val_high > 0: return val_low * (val_high/val_low)**t
    elif val_high > 0: return val_high
    return 0.0

E_co1, E_co2 = 1.1732, 1.3325
rho = 1.0

xs_co1 = {k: interpolate_xlog(E_co1, i+1) for i, k in enumerate(['coherent','incoherent','photoelectric','pair_nuclear','pair_electron','total'])}
xs_co2 = {k: interpolate_xlog(E_co2, i+1) for i, k in enumerate(['coherent','incoherent','photoelectric','pair_nuclear','pair_electron','total'])}

print("Cross sections:")
for E, xs in [(E_co1, xs_co1), (E_co2, xs_co2)]:
    print(f"E={E}: incoherent={xs['incoherent']:.6e}, photo={xs['photoelectric']:.6e}, pair_nuc={xs['pair_nuclear']:.6e}")

mu_tr_over_rho_avg = 0.0
for E, xs_E in [(E_co1, xs_co1), (E_co2, xs_co2)]:
    alpha = E/0.511
    costheta = np.linspace(-1, 1, 100)
    sig_ws, T_E_ws = [], []
    for ct in costheta:
        E_p = E/(1+alpha*(1-ct))
        T = E-E_p
        r0sq = 0.066524e-24/(4*np.pi)
        sig = ((1+ct**2)/2)*(E_p/E)**2*r0sq
        sig_ws.append(sig)
        T_E_ws.append(T/E)
    
    mean_T_over_E = np.sum(np.array(T_E_ws)*np.array(sig_ws))/np.sum(sig_ws)
    print(f"\nE={E}:")
    print(f"  mean_T_over_E = {mean_T_over_E:.4f}")
    
    et_cs = xs_E['incoherent']*mean_T_over_E + xs_E['photoelectric']*1.0
    if E == E_co2:
        et_cs += xs_E['pair_nuclear']*max(0,(E-1.022)/E)
    
    print(f"  Energy transfer cross section = {et_cs:.6e}")
    mu_tr_over_rho_avg += 0.5*et_cs

print(f"\nmu_tr_over_rho_average = {mu_tr_over_rho_avg:.6e} cm^2/g")

L1_primary = 0.729482  # From previous calc
L1_total = L1_primary * 1.10
E_mean_prim = (E_co1+E_co2)/2

print(f"\nL1_primary = {L1_primary}")
print(f"L1_total = {L1_total}")
print(f"E_mean = {E_mean_prim}")

K_MeV_per_g = L1_total * E_mean_prim * mu_tr_over_rho_avg
print(f"\nK (MeV/g) = {L1_total} * {E_mean_prim} * {mu_tr_over_rho_avg:.6e} = {K_MeV_per_g:.6e}")

K_Gy = K_MeV_per_g * 1.602e-10
print(f"K (Gy) = {K_Gy:.6e}")

# Check units: Gy = J/kg = MeV/kg * 1.602e-13... no wait
# Actually 1 MeV = 1.602e-13 J
# 1 Gy = 1 J/kg
# So: MeV/g -> J/kg: multiply by 1.602e-13 J/MeV * 1000 g/kg = 1.602e-10
# Yes, that's correct

print(f"\nExpected order of magnitude: ~10^-12 Gy/cm^2 per fluence")