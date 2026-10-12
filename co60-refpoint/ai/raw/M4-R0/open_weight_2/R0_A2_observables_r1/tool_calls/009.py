import numpy as np

# Load data
xcom = np.loadtxt('/tmp/data/xcom_water.txt')
estar = np.loadtxt('/tmp/data/estar_water.txt')

E_photons = np.array([1.1732, 1.3325])

# Interpolate xcom coefficients
def interp_xcom(E):
    # Assuming cols: E, rayleigh/coherent, incoherent, photoelectric, pair_nuc, pair_el, total, extra
    # Let's map: coherent=col1, incoherent=col2, photo=col3, pair_nuc=col4, pair_el=col5, total=col6
    mu_coherent = np.interp(E, xcom[:,0], xcom[:,1])
    mu_compton = np.interp(E, xcom[:,0], xcom[:,2])
    mu_photo = np.interp(E, xcom[:,0], xcom[:,3])
    mu_pair_nuc = np.interp(E, xcom[:,0], xcom[:,4])
    mu_pair_el = np.interp(E, xcom[:,0], xcom[:,5])
    mu_total = np.interp(E, xcom[:,0], xcom[:,6])
    return mu_coherent, mu_compton, mu_photo, mu_pair_nuc, mu_pair_el, mu_total

mu_c = interp_xcom(1.1732)[0] + interp_xcom(1.3325)[0]
mu_comp = interp_xcom(1.1732)[1] + interp_xcom(1.3325)[1]
mu_pe = interp_xcom(1.1732)[2] + interp_xcom(1.3325)[2]
mu_pn = interp_xcom(1.1732)[3] + interp_xcom(1.3325)[3]
mu_pe2 = interp_xcom(1.1732)[4] + interp_xcom(1.3325)[4]
mu_t = interp_xcom(1.1732)[5] + interp_xcom(1.3325)[5]

print("Mu values per photon (averaged for 50% each line):")
# Actually, we should average the coefficients since lines have equal probability.
coeffs_1 = interp_xcom(1.1732)
coeffs_2 = interp_xcom(1.3325)
mu_avg = [(c1+c2)/2 for c1,c2 in zip(coeffs_1, coeffs_2)]
names = ['coherent', 'compton', 'photo', 'pair_nuc', 'pair_el', 'total']
for n,m in zip(names, mu_avg):
    print(f"{n}: {m}")
