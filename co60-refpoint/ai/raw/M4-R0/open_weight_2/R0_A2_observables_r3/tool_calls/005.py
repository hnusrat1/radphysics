E_tr = [1.0, 1.022, 1.25, 1.5]
T_tr = [7.0514e-2, 6.9769e-2, 6.3061e-2, 5.7400e-2]

def interp_log(x, xdat, ydat):
    return np.exp(np.interp(np.log(x), np.log(xdat), np.log(ydat)))

mu_tr_1 = interp_log(1.1732, E_tr, T_tr)
mu_tr_2 = interp_log(1.3325, E_tr, T_tr)
print(f"mu_tr_1: {mu_tr_1:.6f}, mu_tr_2: {mu_tr_2:.6f}")
print(f"Avg mu_tr: {0.5*(mu_tr_1+mu_tr_2):.6f}")

# K/Phi0 = avg_mu_tr * <E_photon>? No, Kerma K = Psi * (mu_tr/rho). 
# But we are normalizing to Phi0 (fluence). 
# K = Phi0_primary * exp(-mu d) * (mu_tr/rho)_eff? 
# Actually, for a monoenergetic beam, K = Phi * E_gamma * (mu_tr/rho)/E_gamma? No.
# Kerma factor k = (mu_tr/rho). Units: cm^2/g. It represents energy transferred per unit fluence per unit mass? 
# Wait, Kerma K (Gy) = Phi (cm^-2) * E_gamma (MeV) * (mu_tr/rho) (cm^2/g) * 1.602e-10 (J/MeV) * 1000 (g/kg)?
# Actually, standard definition: K = Psi * (mu_tr/rho), where Psi is energy fluence (MeV/cm^2).
# So if we use Fluence Phi0, we need K = Phi0 * <E_avg> * (mu_tr/rho)_weighted.
# Let's compute <E_avg> for Co-60: (1.1732 + 1.3325)/2 = 1.25285 MeV.
E_avg = 1.25285
mu_tr_avg = 0.5 * (interp_log(1.1732, E_tr, T_tr) + interp_log(1.3325, E_tr, T_tr))
K_per_Phi0_MeV_cm2_g = E_avg * mu_tr_avg # This is actually wrong. (mu_tr/rho) has units cm^2/g. Phi0 is cm^-2. Product is cm^0/g? 
# Wait, Kerma K = Psi_E * (mu_tr/rho). Psi_E = Phi * E. So K = Phi * E * (mu_tr/rho). Units: cm^-2 * MeV * cm^2/g = MeV/g. Correct.
K_MeV_per_g_per_Phi0 = E_avg * mu_tr_avg
print(f"K/MuTr product: {K_MeV_per_g_per_Phi0:.6f} MeV/g")

# Convert to Gy: 1 MeV/g = 1.60218e-4 Gy
K_Gy_per_Phi0 = K_MeV_per_g_per_Phi0 * 1.60218e-4
print(f"K_Gy/Phi0: {K_Gy_per_Phi0:.2e}")

# Attenuation factor for primary fluence at 5 cm:
mu_tot_avg = 0.5 * (interp_log(1.1732, [1.0, 1.5], [7.057e-2, 5.7425e-2]) + interp_log(1.3325, [1.0, 1.5], [7.057e-2, 5.7425e-2]))
attenuation = np.exp(-mu_tot_avg * 5.0)
print(f"Attenuation at 5 cm: {attenuation:.4f}")
print(f"L1_primary_fluence_per_phi0: {attenuation:.4f}")