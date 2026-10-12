
import numpy as np

L4 = np.load('L4_data.npz')
Eg = L4['Eg']; Phi = L4['Phi']; Phi_p = L4['Phi_p']
Scol = L4['Scol']; Srest = L4['S_rest']
dE = 0.002
Phi_tot_int = Phi.sum()*dE
Phi_p_int = Phi_p.sum()*dE
print("Phi_total:", Phi_tot_int, "Phi_primary:", Phi_p_int)
print("L4_delta_fraction:", (Phi_tot_int-Phi_p_int)/Phi_tot_int)
# also compute mean energy from phi and rad/coll
Srad = 0.0155*(Eg/1.0)**1.15
print("mean E:", np.sum(Eg*Phi*dE)/Phi_tot_int)
print("rad/coll:", np.sum(Phi*Srad*dE)/np.sum(Phi*Scol*dE))
print("Phi_spec sum:", L4['Phi_spec'].sum())
print("Phi_spec:", np.round(L4['Phi_spec'],6))
