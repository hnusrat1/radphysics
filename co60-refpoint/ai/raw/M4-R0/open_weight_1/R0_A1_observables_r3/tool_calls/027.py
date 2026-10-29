
import numpy as np, pickle

mec2=0.5110
N_e_per_g = 6.02214076e23*10/18.01528
E_tab = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0])
mu_tab = np.array([4.951,1.374,0.7031,0.4470,0.2269,0.2059,0.1837,0.1706,0.1505,0.1370,0.1186,0.1061,0.0969,0.0896,0.0786,0.0707,0.0636,0.0576,0.0494])
def interp(x,xe,ye): return np.interp(x,xe,ye)
def mu_tot(E): return interp(E,E_tab,mu_tab)

E_en = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.4])
muen_tab = np.array([4.57,0.529,0.154,0.0654,0.0410,0.0319,0.0261,0.0255,0.0276,0.0300,0.0319,0.0328,0.0330,0.0329,0.0320,0.0310,0.0303,0.0297,0.0292,0.0290])
def muen(E): return interp(E, E_en, muen_tab)
def mutr(E): return muen(E)*(1+0.0055)

ph_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ph_c = (ph_bins[:-1]+ph_bins[1:])/2
prim_spec = np.load('prim_spec.npy')
scat_spec = np.load('scat_spec.npy')
tot_spec = prim_spec+scat_spec

Cg = 1.602e-10
K = np.sum(tot_spec*mutr(ph_c)*ph_c)*1.602e-10
Kcol = np.sum(tot_spec*muen(ph_c)*ph_c)*1.602e-10
Kcol_scat = np.sum(scat_spec*muen(ph_c)*ph_c)*1.602e-10
print("K/Phi0:", K, " Kcol/Phi0:", Kcol)
print("g = 1-Kcol/K:", 1-Kcol/K)
print("Kcol scattered/total:", Kcol_scat/Kcol)
print("D/Phi0 (beta=1):", Kcol)

Emean = np.sum((prim_spec+scat_spec)*ph_c)/np.sum(prim_spec+scat_spec)
print("mean photon energy:", Emean)
print("scattered energy fluence fraction:", np.sum(scat_spec*ph_c)/np.sum((prim_spec+scat_spec)*ph_c))

with open('photon_mc.pkl','rb') as f:
    data = pickle.load(f)
res = data['res']; z0s = data['z0s']; N = data['N']
phi0ph = 1/(0.01*105**2)
Kcol_z = np.zeros(len(z0s))
for i in range(len(z0s)):
    ps = res['prim_spec'][i]/(N*phi0ph)
    ss = res['scat_spec'][i]/(N*phi0ph)
    Kcol_z[i] = np.sum((ps+ss)*muen(ph_c)*ph_c)*1.602e-10
for i,z0 in enumerate(z0s):
    print(f"z={z0:5.1f} Kcol/Phi0={Kcol_z[i]:.5e}")
i05 = np.where(z0s==0.5)[0][0]
i5 = np.where(z0s==5.0)[0][0]
print("PDD5 = D(5)/D(0.5):", 100*Kcol_z[i5]/Kcol_z[i05])

# s_w_air
L4 = np.load('L4_data.npz')
Eg = L4['Eg']; Phi = L4['Phi']; S_rest = L4['S_rest']; Scol = L4['Scol']
dE=0.002
n_e_air = 3.01e23
Scol_air = np.interp(Eg, [0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.5,0.7,1.0,1.25,1.4],
                     [19.6,9.65,6.87,5.30,4.30,3.66,2.30,1.90,1.84,1.79,1.765,1.75,1.75])
delta_loss_w = Scol - S_rest
S_rest_air = Scol_air - delta_loss_w*(n_e_air/3.343e23)
Phi_delta_val = Phi[np.argmin(np.abs(Eg-0.010))]
num = np.sum(Phi*S_rest)*dE + Phi_delta_val*0.010
den = np.sum(Phi*S_rest_air)*dE + Phi_delta_val*0.010
print("s_w_air (SA, Delta=10 keV):", num/den)

prim_ang = np.load('prim_ang.npy'); scat_ang = np.load('scat_ang.npy')
print("primary angular (frac):", prim_ang/np.maximum(prim_ang.sum(),1e-30))
print("scattered angular (frac):", np.round(scat_ang/np.sum(scat_ang),5))

# L4 mean energy and rad/coll from saved data
mean_e = np.sum(Eg*Phi*dE)/np.sum(Phi*dE)
Srad = 0.0155*(Eg/1.0)**1.15
print("L4 mean electron energy:", mean_e)
print("L4 rad/coll:", np.sum(Phi*Srad*dE)/np.sum(Phi*Scol*dE))
print("L4 electron fluence/Phi0:", np.sum(Phi*dE))
