import numpy as np
r=np.load('/tmp/test_photon_mc.npy',allow_pickle=True).item()
N=r['N']; Omega105_2=109.97517633116931; Vcyl=np.pi*0.2
print("Vcyl",Vcyl)
z=r['zc_slabs']; hh=r['hh_slab']; Vslab=np.pi*hh*2  # radius 1 -> V=pi*1^2*(2hh)
# primary fluence per Phi0 vs depth
for k in [0,2,4,6,8,10,12,16,20,24,28,31]:
    f1 = r['slab_prim'][0,k]/N / Vslab * Omega105_2
    f2 = r['slab_prim'][1,k]/N / Vslab * Omega105_2
    ftotp = (r['slab_prim'][0,k]+r['slab_prim'][1,k])/N/Vslab*Omega105_2
    exp1=0.5*np.exp(-0.06514178*float(z[k])); exp2=0.5*np.exp(-0.06104757*float(z[k]))
    print(f"z={z[k]:.2f}: prim1/ph0={f1:.4f} (exp {exp1:.4f}) prim2={f2:.4f} (exp {exp2:.4f}) totprim={ftotp:.4f}")
# scattered fluence per Phi0
fsec = r['slab_sec'][:, :].sum(axis=1)/N/Vslab*Omega105_2
fprim=(r['slab_prim'][0]+r['slab_prim'][1])/N/Vslab*Omega105_2
print("\nscattered fluence/ph0 at depths:", np.round(fsec[[0,2,4,6,8,10,12,16,20]],4))
# Kcol per Phi0 (MeV/g) from muenE scoring
Kcol_MeVg = r['slab_Kcol']/N/Vslab*Omega105_2
print("Kcol(MeV/g per Phi0):", np.round(Kcol_MeVg[[0,2,4,6,8,10]],4))
print("Kcol[Gy cm^2] at z=5:", Kcol_MeVg[20]*1.602e-10)
# primary Kcol at depth z: phi(z)*muenE_avg
import scipy.interpolate as si
E1=1.1732; E2=1.3325
# recompute muenE for the two lines quickly (approx values from earlier)
muenE1=0.03519; muenE2=0.03882
Kprim_5= (0.5*np.exp(-0.06514*5)*muenE1 + 0.5*np.exp(-0.06105*5)*muenE2)
print("analytic primary Kcol at 5 cm (MeV/g per Phi0):",Kprim_5, "Gycm2:",Kprim_5*1.602e-10)