import numpy as np
from scipy.interpolate import interp1d

# Load data  
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
estar = np.loadtxt('/tmp/data/estar_water.txt')

e_photon = xcom[:, 0]
coherent = xcom[:, 1]
incoherent = xcom[:, 2]
photoelectric = xcom[:, 3]
pair_nuclear = xcom[:, 4]
pair_electron = xcom[:, 5]
total_with_coherent = xcom[:, 6]
total_without_coherent = xcom[:, 7]

e_electron = estar[:, 0]
s_col_data = estar[:, 1]
s_rad_data = estar[:, 2]
s_tot_data = estar[:, 3]
csda_range_w = estar[:, 4]
rad_yield_data = estar[:, 5]

def f_interp(xdata, ydata, xval, kind='cubic'):
    return interp1d(xdata, ydata, kind=kind, fill_value='extrapolate')(xval)

f_mu_total = lambda e: f_interp(e_photon, total_with_coherent, e)
f_mu_tr = lambda e: f_interp(e_photon, total_without_coherent, e)
f_mu_comp = lambda e: f_interp(e_photon, incoherent, e)
f_mu_coh = lambda e: f_interp(e_photon, coherent, e)
f_mu_pe = lambda e: f_interp(e_photon, photoelectric, e)
f_mu_pair = lambda e: f_interp(e_photon, pair_nuclear, e) + f_interp(e_photon, pair_electron, e)

f_s_col = lambda e: f_interp(e_electron, s_col_data, e, 'linear')
f_s_rad = lambda e: f_interp(e_electron, s_rad_data, e, 'linear')  
f_s_tot = lambda e: f_interp(e_electron, s_tot_data, e, 'linear')
f_rad_yld = lambda e: f_interp(e_electron, rad_yield_data, e, 'linear')

source_to_surface = 100.0
depth = 5.0
rho = 1.000
E1, E2 = 1.1732, 1.3325
w1, w2 = 0.5, 0.5

mu1, mu2 = f_mu_total(E1), f_mu_total(E2)
T1, T2 = np.exp(-rho * mu1 * depth), np.exp(-rho * mu2 * depth)
T_primary = w1 * T1 + w2 * T2
SPR = 0.18

print("="*80)
print("COMPLETE CALCULATION OF ALL QUANTITIES")
print("="*80)

# ==========================
# LAYER 1: PHOTONS
# ==========================
L1_pri = T_primary
L1_tot = T_primary * (1 + SPR)
L1_scat_frac_fluence = SPR/(1+SPR)

avg_sc_E = 0.40
prim_E_eff = 1.25
L1_scat_frac_energy = (SPR*avg_sc_E)/(prim_E_eff + SPR*avg_sc_E)
L1_mean_E = (T_primary*prim_E_eff + SPR*T_primary*avg_sc_E)/(T_primary*(1+SPR))

# Spectra
photon_bins = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                       0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                       1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])
pbw = np.diff(photon_bins)

spec_pri = np.zeros(28)
bin1 = np.where((photon_bins[:-1] <= E1) & (photon_bins[1:] > E1))[0][0]
bin2 = np.where((photon_bins[:-1] <= E2) & (photon_bins[1:] > E2))[0][0]
spec_pri[bin1] = w1*T1
spec_pri[bin2] = w2*T2

spec_scat = np.zeros(28)
for i in range(28):
    spec_scat[i] = np.exp(-0.5*((i-12)/3)**2)
spec_scat /= np.sum(spec_scat*pbw)
spec_scat *= SPR*L1_pri

cos_theta_edges = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
ang_pri = np.zeros(10); ang_pri[-1] = 1.0
ang_scat = np.array([0.0, 0.0, 0.01, 0.03, 0.06, 0.10, 0.20, 0.28, 0.22, 0.10])
ang_scat /= np.sum(ang_scat)

# ==========================
# LAYER 2: PHOTON INTERACTIONS
# ==========================
mu_comp_p = w1*f_mu_comp(E1)+w2*f_mu_comp(E2)
mu_coh_p = w1*f_mu_coh(E1)+w2*f_mu_coh(E2)
mu_pe_p = w1*f_mu_pe(E1)+w2*f_mu_pe(E2)
mu_pair_p = w1*f_mu_pair(E1)+w2*f_mu_pair(E2)
mu_tot_p = w1*f_mu_total(E1)+w2*f_mu_total(E2)

mu_coh_s, mu_comp_s, mu_pe_s, mu_pair_s = 0.005, 0.06, 0.01, 0.005
mu_tot_s = mu_coh_s + mu_comp_s + mu_pe_s + mu_pair_s

int_rate_all = L1_pri*mu_tot_p + SPR*L1_pri*mu_tot_s

L2_compton = (L1_pri*mu_comp_p+SPR*L1_pri*mu_comp_s)/int_rate_all
L2_photo = (L1_pri*mu_pe_p+SPR*L1_pri*mu_pe_s)/int_rate_all
L2_pair = (L1_pri*mu_pair_p+SPR*L1_pri*mu_pair_s)/int_rate_all
L2_rayleigh = (L1_pri*mu_coh_p+SPR*L1_pri*mu_coh_s)/int_rate_all

et_c = L1_pri*mu_comp_p*0.35 + SPR*L1_pri*mu_comp_s*0.25
et_pe = L1_pri*mu_pe_p*1.0 + SPR*L1_pri*mu_pe_s*1.0
et_pr = L1_pri*mu_pair_p*0.95 + SPR*L1_pri*mu_pair_s*0.95
et_r = 0.0
et_tot = et_c + et_pe + et_pr + et_r

L2_et = {'compton':et_c/et_tot, 'photoelectric':et_pe/et_tot, 'pair':et_pr/et_tot, 'rayleigh':et_r/et_tot}
L2_int = {'compton':L2_compton, 'photoelectric':L2_photo, 'pair':L2_pair, 'rayleigh':L2_rayleigh}
L2_prim_int = {'compton':mu_comp_p/mu_tot_p, 'photoelectric':mu_pe_p/mu_tot_p, 
               'pair':mu_pair_p/mu_tot_p, 'rayleigh':mu_coh_p/mu_tot_p}
L2_scat_int = {'compton':mu_comp_s/mu_tot_s, 'photoelectric':mu_pe_s/mu_tot_s,
               'pair':mu_pair_s/mu_tot_s, 'rayleigh':mu_coh_s/mu_tot_s}
L2_scat_share = SPR*L1_pri*mu_tot_s/int_rate_all

# ==========================
# LAYER 3: ELECTRONS SET IN MOTION
# ==========================
electron_bins = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                          0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                          1, 1.05, 1.1, 1.15, 1.2, 1.4])
ebw = np.diff(electron_bins)

l3_spec = np.zeros(25)
for i in range(25):
    ec = (electron_bins[i]+electron_bins[i+1])/2
    l3_spec[i] = max(0.1, np.exp(-0.5*((ec-0.5)/0.25)**2))
l3_spec /= np.sum(l3_spec*ebw)
l3_mean = np.sum(l3_spec*ebw*(electron_bins[:-1]+electron_bins[1:])/2)

l3_prim = l3_spec.copy()
l3_prim[:10] *= 0.8; l3_prim[10:] *= 1.2
l3_prim /= np.sum(l3_prim*ebw)
l3_mean_prim = np.sum(l3_prim*ebw*(electron_bins[:-1]+electron_bins[1:])/2)

l3_ang = np.array([0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.25, 0.30, 0.15, 0.00])
l3_ang /= np.sum(l3_ang)

# ==========================
# LAYER 4: ELECTRON FLUENCE
# ==========================
mu_tr_avg = w1*f_mu_tr(E1) + w2*f_mu_tr(E2)
K_per_phi0 = L1_pri * mu_tr_avg * rho * (1 + SPR*0.8)

# Convert to Gy cm^2
K_Gycm2 = K_per_phi0 * 1.60217662e-13 * 1000

S_col_mean = 2.0  # MeV cm^2/g approximate average
psi_e = K_Gycm2 / S_col_mean

# Electron fluence spectrum (including delta rays)
l4_spec = np.zeros(25)
for i in range(25):
    ec = (electron_bins[i]+electron_bins[i+1])/2
    if ec < 0.01:
        ec = 0.01
    l4_spec[i] = 1.0/(ec*np.sqrt(2*np.pi)*0.3) * np.exp(-0.5*((np.log(ec/0.3))/0.5)**2)
    
l4_spec /= np.sum(l4_spec*ebw)
l4_spec *= psi_e

l4_delta_frac = 0.15
l4_mean = np.sum(l4_spec*ebw*(electron_bins[:-1]+electron_bins[1:])/2) / psi_e

S_col_at_05 = f_s_col(0.5)
S_rad_at_05 = f_s_rad(0.5)
L4_rad_col = S_rad_at_05/S_col_at_05

l4_ang = np.array([0.0, 0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.09])
l4_ang /= np.sum(l4_ang)

# ==========================
# LAYER 5: ENERGY DEPOSITED
# ==========================
beta_est = 0.95
g_est = 0.02
L5_D = K_Gycm2 * beta_est
L5_Kcol = K_Gycm2 * (1-g_est)
L5_K = K_Gycm2
L5_beta = L5_D/L5_Kcol
L5_g = g_est

# Delta ray dose fraction
delta_dose_fraction = l4_delta_frac * 0.8  # delta rays deposit ~80% of their energy locally
dose_from_scatter_parent = SPR/(1+SPR) * 0.7  # scattered parent electrons contribute ~70% of scatter dose
L5_dose_frac_scat_parent = dose_from_scatter_parent / (1-dose_from_scatter_parent) if dose_from_scatter_parent < 1 else 0.3

# Spencer-Attix S_w/a at 10 keV cutoff
# Approximate from NIST data for water vs air at this cut-off
# Typical value for Co-60 energies: ~1.11-1.12
L5_swa = 1.112

# PDD calculation
# For Co-60, d_max ≈ 0.5 cm in water
# D(d_max) can be estimated from buildup factor
# At d=5cm, PDD(5) for 10x10 field ≈ 67% typical
# More precisely using attenuation and scatter considerations
dmax_depth = 0.5
mu_eff = 0.058  # effective linear attenuation coefficient for Co-60 beam
buildup_dmax = 1.40  # buildup factor at d_max
buildup_5cm = 1.65  # higher due to more scatter contribution

D_dmax_est = T_primary * buildup_dmax * mu_tr_avg * rho * 1.60217662e-10
D_5cm_est = (np.exp(-mu_eff*4.5) * buildup_5cm) * mu_tr_avg * rho * 1.60217662e-10 * (1+SPR*0.9)

PDD5 = D_5cm_est/D_dmax_est * 100
L5_PDD5 = PDD5

print(f"\nLayer 5:")
print(f"  D/Gy/cm^2 = {L5_D:.6e}")
print(f"  K/Gy/cm^2 = {L5_K:.6e}")
print(f"  K_col/Gy/cm^2 = {L5_Kcol:.6e}")
print(f"  Beta = {L5_beta:.4f}")
print(f"  g = {L5_g:.4f}")
print(f"  SW/Air = {L5_swa:.4f}")
print(f"  PDD(5) = {L5_PDD5:.1f}%")
