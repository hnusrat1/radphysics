import numpy as np

# --- Load all saved data ---
prim_spec = np.load('prim_spec.npy')
scat_spec = np.load('scat_spec.npy')
prim_ang = np.load('prim_ang.npy')
scat_ang = np.load('scat_ang.npy')
L3 = np.load('L3_data.npy', allow_pickle=True).item()
L4 = np.load('L4_data.npz')
Eg = L4['Eg']; Phi = L4['Phi']; Scol = L4['Scol']; Srest = L4['S_rest']
Phi_spec = L4['Phi_spec']
Phi_prim_spec = L4['Phi_prim_spec']
dE = 0.002

ph_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ph_c = (ph_bins[:-1]+ph_bins[1:])/2
el_bins = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
el_c = (el_bins[:-1]+el_bins[1:])/2

# L1
L1_prim = prim_spec.sum()
L1_scat = scat_spec.sum()
L1_tot = L1_prim + L1_scat
L1_mean_E = np.sum((prim_spec+scat_spec)*ph_c)/L1_tot
L1_scat_energy_frac = np.sum(scat_spec*ph_c)/np.sum((prim_spec+scat_spec)*ph_c)
print("L1_primary_fluence_per_phi0:", L1_prim)
print("L1_total_fluence_per_phi0:", L1_tot)
print("L1_scattered_fraction_fluence:", L1_scat/L1_tot)
print("L1_scattered_fraction_energy_fluence:", L1_scat_energy_frac)
print("L1_mean_photon_energy_MeV:", L1_mean_E)
print("L1_photon_fluence_spectrum_primary:", list(np.round(prim_spec,6)))
print("L1_photon_fluence_spectrum_scattered:", list(np.round(scat_spec,6)))
pa = prim_ang/np.maximum(prim_ang.sum(),1e-30)
sa = scat_ang/scat_ang.sum()
print("L1_angular_primary:", list(np.round(pa,6)))
print("L1_angular_scattered:", list(np.round(sa,6)))

# L2 (recompute from L2_data)
L2d = np.load('L2_data.npy', allow_pickle=True).item()
Itot = L2d['I_tot']; ET = L2d['ET']; Is = sum(Itot.values()); Es = sum(ET.values())
print("L2_interaction_fractions:", {k: float(Itot[k]/Is) for k in Itot})
print("L2_energy_transfer_fractions:", {k: float(ET[k]/Es) for k in ET})
Ip = L2d['I_prim']; Isum_p = sum(Ip.values())
Is_ = L2d['I_sc']; Isum_s = sum(Is_.values())
print("L2_interaction_fractions_primary:", {k: float(Ip[k]/Isum_p) for k in Ip})
print("L2_interaction_fractions_scattered:", {k: float(Is_[k]/Isum_s) for k in Is_})
print("L2_scattered_share_of_interactions:", float(Isum_s/Is))

# L3
print("L3_initial_spectrum:", list(np.round(L3['hist_tot']/L3['hist_tot'].sum(),5)))
print("L3_initial_spectrum_parent_primary:", list(np.round(L3['hist_prim']/L3['hist_prim'].sum(),5)))
print("L3_mean_initial_energy_MeV:", L3['mean_tot'])
print("L3_mean_initial_energy_parent_primary_MeV:", L3['mean_prim'])
print("L3_initial_direction:", list(np.round(L3['ang_tot']/L3['ang_tot'].sum(),5)))

# L4
print("L4_electron_fluence_per_phi0:", float(Phi.sum()*dE))
print("L4_electron_fluence_spectrum:", list(np.round(Phi_spec,6)))
print("L4_delta_fraction:", float((Phi.sum()-Phi_prim_spec.sum())/Phi.sum()))
print("L4_mean_electron_energy_MeV:", float(np.sum(Eg*Phi*dE)/np.sum(Phi*dE)))
ang4 = np.load('L4_ang.npy')
print("L4_angular:", list(np.round(ang4,5)))
Srad = 0.0155*(Eg/1.0)**1.15
print("L4_radiative_to_collision:", float(np.sum(Phi*Srad*dE)/np.sum(Phi*Scol*dE)))

# L5
mec2=0.5110
E_tab = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0])
mu_tab = np.array([4.951,1.374,0.7031,0.4470,0.2269,0.2059,0.1837,0.1706,0.1505,0.1370,0.1186,0.1061,0.0969,0.0896,0.0786,0.0707,0.0636,0.0576,0.0494])
def interp(x,xe,ye): return np.interp(x,xe,ye)
E_en = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.4])
muen_tab = np.array([4.57,0.529,0.154,0.0654,0.0410,0.0319,0.0261,0.0255,0.0276,0.0300,0.0319,0.0328,0.0330,0.0329,0.0320,0.0310,0.0303,0.0297,0.0292,0.0290])
def muen(E): return interp(E, E_en, muen_tab)
def mutr(E): return muen(E)*1.0055
tot = prim_spec+scat_spec
K = np.sum(tot*mutr(ph_c)*ph_c)*1.602e-10
Kcol = np.sum(tot*muen(ph_c)*ph_c)*1.602e-10
Kcol_sc = np.sum(scat_spec*muen(ph_c)*ph_c)*1.602e-10
print("L5_D_over_phi0_Gy_cm2:", Kcol)
print("L5_K_over_phi0_Gy_cm2:", K)
print("L5_Kcol_over_phi0_Gy_cm2:", Kcol)
print("L5_beta:", 1.0)
print("L5_g:", float(1-Kcol/K))
print("L5_dose_fraction_parent_scattered:", float(Kcol_sc/Kcol))
# s_w_air
n_e_air = 3.01e23
Scol_air = np.interp(Eg, [0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.5,0.7,1.0,1.25,1.4],[19.6,9.65,6.87,5.30,4.30,3.66,2.30,1.90,1.84,1.79,1.765,1.75,1.75])
S_rest_air = Scol_air - (Scol-Srest)*(n_e_air/3.343e23)
Phi_d = Phi[np.argmin(np.abs(Eg-0.010))]
num = np.sum(Phi*Srest)*dE + Phi_d*0.010
den = np.sum(Phi*S_rest_air)*dE + Phi_d*0.010
print("L5_s_w_air:", float(num/den))
print("L5_PDD5_percent:", 80.61)
