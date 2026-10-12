import numpy as np

# load xcom
d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_ph = d[:,0]; mu_coh=d[:,1]; mu_incoh=d[:,2]; mu_pe=d[:,3]; mu_pairN=d[:,4]; mu_pairE=d[:,5]; mu_tot=d[:,6]
print("xcom energy range", E_ph.min(), E_ph.max(), "n=",len(E_ph))
for e in [0.2,0.5,1.0,1.1732,1.25,1.3325,2.0,5.0]:
    i=np.argmin(np.abs(E_ph-e))
    print(f"E={E_ph[i]:.4g}: tot={mu_tot[i]:.5g} coh={mu_coh[i]:.4g} inco={mu_incoh[i]:.5g} pe={mu_pe[i]:.4g} pairN={mu_pairN[i]:.4g} pairE={mu_pairE[i]:.4g}")

# load estar
d2 = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=d2[:,0]; Sc=d2[:,1]; Sr=d2[:,2]; St=d2[:,3]; R=d2[:,4]; Y=d2[:,5]; de=d2[:,6]
print("\nestar range", Ee.min(), Ee.max(), "n=", len(Ee))
for e in [0.01,0.1,0.2,0.5,1.0,1.1732,1.3325,2.0,10.0]:
    i=np.argmin(np.abs(Ee-e))
    print(f"E={Ee[i]:.4g}: Sc={Sc[i]:.5g} Sr={Sr[i]:.5g} St={St[i]:.5g} R={R[i]:.5g} Y={Y[i]:.5g}")