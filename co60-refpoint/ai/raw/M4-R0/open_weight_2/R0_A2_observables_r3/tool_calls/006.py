import numpy as np

E_tr = [1.0, 1.022, 1.25, 1.5]
T_tr = [7.0514e-2, 6.9769e-2, 6.3061e-2, 5.7400e-2]

def interp_log(x, xdat, ydat):
    return np.exp(np.interp(np.log(x), np.log(xdat), np.log(ydat)))

mu_tr_1 = interp_log(1.1732, E_tr, T_tr)
mu_tr_2 = interp_log(1.3325, E_tr, T_tr)
print(f"mu_tr_1: {mu_tr_1:.6f}, mu_tr_2: {mu_tr_2:.6f}")
mu_tr_avg = 0.5 * (mu_tr_1 + mu_tr_2)

E_avg = 1.25285
K_MeV_per_g_per_Phi0 = E_avg * mu_tr_avg
K_Gy_per_Phi0 = K_MeV_per_g_per_Phi0 * 1.60218e-4

mu_tot_avg = 0.5 * (interp_log(1.1732, [1.0, 1.5], [7.057e-2, 5.7425e-2]) + interp_log(1.3325, [1.0, 1.5], [7.057e-2, 5.7425e-2]))
attenuation = np.exp(-mu_tot_avg * 5.0)

print(f"L1_primary_fluence_per_phi0: {attenuation:.4f}")
print(f"K_Gy/Phi0: {K_Gy_per_Phi0:.2e}")
print(f"Avg mu_tr: {mu_tr_avg:.6f}")