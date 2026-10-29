import numpy as np

def parse_xcom(path):
    lines = open(path).readlines()
    E = []
    coher = []
    incont = []
    photo = []
    pair_nuc = []
    pair_elec = []
    total = []
    
    for line in lines:
        if line.startswith('#'): continue
        parts = line.split()
        if len(parts) >= 8:
            try:
                E.append(float(parts[0]))
                coher.append(float(parts[1]))
                incont.append(float(parts[2]))
                photo.append(float(parts[3]))
                pair_nuc.append(float(parts[4]))
                pair_elec.append(float(parts[5]))
                total.append(float(parts[6]))
            except: pass
    return np.array(E), np.array(coher), np.array(incont), np.array(photo), np.array(pair_nuc), np.array(pair_elec), np.array(total)

E_xc, coh, inc, pho, pn, pe, tot = parse_xcom('/tmp/data/xcom_water.txt')

# Interpolate for Co-60 energies
E1, E2 = 1.1732, 1.3325
def interp_val(x, xdat, ydat):
    return np.interp(x, xdat, ydat)

mu_tot_1 = interp_val(E1, E_xc, tot)
mu_tot_2 = interp_val(E2, E_xc, tot)
mu_inc_1 = interp_val(E1, E_xc, inc)
mu_inc_2 = interp_val(E2, E_xc, inc)
mu_pho_1 = interp_val(E1, E_xc, pho)
mu_pho_2 = interp_val(E2, E_xc, pho)
mu_pn_1 = interp_val(E1, E_xc, pn)
mu_pn_2 = interp_val(E2, E_xc, pn)
mu_pe_1 = interp_val(E1, E_xc, pe)
mu_pe_2 = interp_val(E2, E_xc, pe)
mu_coh_1 = interp_val(E1, E_xc, coh)
mu_coh_2 = interp_val(E2, E_xc, coh)

print(f"Mu_tot: {mu_tot_1:.4f}, {mu_tot_2:.4f}")
print(f"Mu_inc: {mu_inc_1:.4f}, {mu_inc_2:.4f}")
print(f"Mu_pho: {mu_pho_1:.4f}, {mu_pho_2:.4f}")
print(f"Mu_pair_nuc: {mu_pn_1:.4f}, {mu_pn_2:.4f}")
print(f"Mu_pair_elec: {mu_pe_1:.4f}, {mu_pe_2:.4f}")
print(f"Mu_coh: {mu_coh_1:.4f}, {mu_coh_2:.4f}")

# Average mu/rho for Co-60 (equal prob)
mu_avg = 0.5 * (mu_tot_1 + mu_tot_2)
print(f"Avg mu/rho: {mu_avg:.4f} cm2/g")