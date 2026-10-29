import numpy as np

# Load data
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')

E_co = np.array([1.1732, 1.3325])
w_co = np.array([0.5, 0.5])
depth = 5.0
rho = 1.0

def interp_x(ci, E):
    return np.interp(np.log10(E), np.log10(xcom[:, 0]), xcom[:, ci])

mu_t_w = sum(w * interp_x(6, E) for w, E in zip(w_co, E_co))
mu_l = mu_t_w * rho
primary_frac_val = np.exp(-mu_l * depth)
print(f"Primary transmission: {primary_frac_val:.6f}")
