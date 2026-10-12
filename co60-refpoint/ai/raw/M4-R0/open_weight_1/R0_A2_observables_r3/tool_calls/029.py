import numpy as np
fl=np.load('fluence_results.npz',allow_pickle=True)
EG=fl['EG']; Phi_all=fl['Phi_all']; Phi_scat=fl['Phi_scat']; Phi_prim=fl['Phi_prim']
sel=[0.6,0.7,0.8,0.85,0.9,0.95,0.975,1.0,1.05,1.075,1.1,1.15,1.2,1.25,1.3,1.35]
for E in sel:
    i=np.argmin(abs(EG-E))
    print(f"E={EG[i]:.4f}: Phi_all={Phi_all[i]:.5e} Phi_prim={Phi_prim[i]:.5e} Phi_scat={Phi_scat[i]:.5e} S0?")
# check L4 spec with finer method: integrate over all grid points in bin (not masked)
EL=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
             0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
def flu_bins(EG,Phi,el):
    out=np.zeros(len(el)-1)
    idx=np.searchsorted(EG,el,side='left')
    for i in range(len(el)-1):
        a,b=idx[i],idx[i+1]
        if b>a+1:
            out[i]=np.trapezoid(Phi[a:b],EG[a:b])
        elif b>a:
            out[i]=Phi[a]*(EG[a+1]-EG[a-1])/2 if a+1<len(EG) else 0
    return out
L4=flu_bins(EG,Phi_all,EL)
print("L4 full:",np.array2string(L4,precision=5))
print("sum:",L4.sum())