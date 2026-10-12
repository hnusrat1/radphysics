import numpy as np

# Load XCOM water data
with open('/tmp/data/xcom_water.txt') as f:
    lines = [l for l in f if not l.startswith('#') and l.strip()]
# assume columns: energy, coherent, incoherent, photoelectric, pair_nuc, pair_elec, total
x = np.loadtxt(lines)
E_x = x[:,0]
mu_coh = x[:,1]; mu_incoh = x[:,2]; mu_pe = x[:,3]; mu_pairn = x[:,4]; mu_paire = x[:,5]; mu_tot = x[:,6]

print("XCOM energies:", E_x[:5], E_x[-5:])
print("mu_tot at 1.17,1.33:", np.interp(1.1732,E_x,mu_tot), np.interp(1.3325,E_x,mu_tot))
print("mu_incoh:", np.interp(1.1732,E_x,mu_incoh), np.interp(1.3325,E_x,mu_incoh))
print("mu_coh:", np.interp(1.1732,E_x,mu_coh))
print("mu_pe:", np.interp(1.1732,E_x,mu_pe))
print("mu_pair tot:", np.interp(1.1732,E_x,mu_pairn+mu_paire), np.interp(1.3325,E_x,mu_pairn+mu_paire))

# Load ESTAR
with open('/tmp/data/estar_water.txt') as f:
    lines2 = [l for l in f if not l.startswith('#') and l.strip()]
est = np.loadtxt(lines2)
Ee = est[:,0]; Scol=est[:,1]; Srad=est[:,2]; Stot=est[:,3]; csda=est[:,4]; yld=est[:,5]; den=est[:,6]
print("\nESTAR first rows:")
print(est[:5])
print("Scol at 1 MeV:", np.interp(1.0,Ee,Scol))
print("yield at 1 MeV:", np.interp(1.0,Ee,yld))
print("csda at 1 MeV:", np.interp(1.0,Ee,csda))