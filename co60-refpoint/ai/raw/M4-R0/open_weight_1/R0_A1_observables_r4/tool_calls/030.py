
import numpy as np
d = np.load('mc5.npz')
Hp = d['Hp'].reshape(28,10,2); Hcnt=d['Hcnt']; Ntot=d['Ntot']
Vcyl=np.pi*0.2
sp_p=Hp[:,:,0].sum(axis=1)/Vcyl/Ntot*110.25
sp_s=Hp[:,:,1].sum(axis=1)/Vcyl/Ntot*110.25
print("sp_p =", np.round(sp_p,6).tolist())
print("sp_s =", np.round(sp_s,6).tolist())
print("sum sp_p", sp_p.sum(), "sum sp_s", sp_s.sum())

W_all=np.load('l3_bins_all.npy'); W_prim=np.load('l3_bins_prim.npy')
print("L3 all:", np.round(W_all/W_all.sum(),5).tolist())
print("L3 prim:", np.round(W_prim/W_prim.sum(),5).tolist())
print("means", (W_all*0.5*(np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2])+np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4]))/2).sum()/W_all.sum())
