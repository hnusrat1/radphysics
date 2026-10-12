
code = r'''
import numpy as np
me = 0.51099895
NA_e = 3.3428e23
N_mol = 3.3428e22
r_e = 2.8179403e-13
MU_PAIR_CM = 0.0001
MU_PE_CM = 0.0

def kn_sigma(E):
    k = E/me
    return 2*np.pi*r_e**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k) + np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2)

def rayleigh_sigma(lam_A):
    a = 0.3
    th = np.linspace(1e-9, np.pi, 2000001)
    s = np.sin(th/2)/lam_A
    F2 = 100/(1+(s/a)**2)**4
    f = 0.5*(1+np.cos(th)**2)*F2*2*np.pi*np.sin(th)
    return r_e**2*np.trapezoid(f, th)

energies = [1.1732, 1.3325]
for E in energies:
    sig_inc = kn_sigma(E)
    lam = 12.398/(E*1000)
    sig_coh = rayleigh_sigma(lam)
    print(E, "inc", NA_e*sig_inc, "coh", N_mol*sig_coh, "sig_inc_b", sig_inc/1e-24, "coh_sig_b", sig_coh/1e-24)
'''
open('xs.py','w').write(code)
import subprocess
print(subprocess.run(['python','xs.py'],capture_output=True,text=True))
