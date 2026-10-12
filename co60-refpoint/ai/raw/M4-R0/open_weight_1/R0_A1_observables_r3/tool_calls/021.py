
import numpy as np
d = np.load('L4_data.npz')
Eg = d['Eg']; Phi = d['Phi']; Phi_p = d['Phi_p']; q = d['q']
S_rest = d['S_rest']; Scol = d['Scol']; Srad = d['Srad']
dE = 0.002
# energy checks
Kcol_elec = np.sum(q*Eg)*dE   # MeV/g per Phi0 (electrons only source, no positrons)
print("energy injected in electrons (per Phi0, MeV/g):", Kcol_elec)
D_elec_rest = np.sum(Phi*S_rest)*dE
TE = Phi[np.argmin(np.abs(Eg-0.010))]*0.010  # track end approx
print("D via Phi*S_rest + TE:", D_elec_rest, "+", TE, "=", D_elec_rest+TE)
# energy transferred to deltas
# recompute q_sec from Phi (single pass)
mec2=0.5110
N_e_per_g=6.02214076e23*10/18.01528
n_e=N_e_per_g
C=0.4987e-24*0.511
def moler_dsde(T,E0):
    if T<=0.011 or E0<=0.010 or E0>=T/2: return 0.0
    beta2=1.0-1.0/(1+T/mec2)**2
    if beta2<=0: return 0.0
    return C*n_e/beta2*(1/E0**2 - 1/(E0*(T-E0)) + 1/(T-E0)**2)
qsec=np.zeros(len(Eg))
start=np.searchsorted(Eg,0.010,side='right')
for ite in range(start,len(Eg)):
    T=Eg[ite]
    if T<0.012: continue
    max_idx=np.searchsorted(Eg,T/2,side='right')-1
    for ie in range(start,max_idx+1):
        E0=Eg[ie]
        if E0>=T/2-1e-12: continue
        dsd=moler_dsde(T,E0)
        if dsd>0: qsec[ie]+=Phi[ite]*dsd*dE
print("energy to deltas (q_sec*E):", np.sum(qsec*Eg)*dE)
print("Scol-Srest energy from Phi:", np.sum(Phi*(Scol-S_rest))*dE)
print("delta energy fraction of injected:", np.sum(qsec*Eg)*dE/Kcol_elec)
print("delta track / primary track:", (Phi.sum()-Phi_p.sum())*dE/(Phi_p.sum()*dE))
print("q_sec integral:", qsec.sum()*dE, " vs Phi_total-Phi_p integrated:", (Phi.sum()-Phi_p.sum())*dE)
