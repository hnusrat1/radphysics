import numpy as np

# Load photon MC results at 5 cm
d = np.load('mc5.npz')
Hp = d['Hp'].reshape(28,10,2)
Hcnt = d['Hcnt']; Ntot = d['Ntot']
Vcyl = np.pi*0.2
sp_p = Hp[:,:,0].sum(axis=1)/Vcyl/Ntot*110.25
sp_s = Hp[:,:,1].sum(axis=1)/Vcyl/Ntot*110.25
fp = sp_p.sum(); fs = sp_s.sum()
print('primary fluence', fp, 'scattered', fs, 'total', fp+fs)
print('scattered fraction fluence', fs/(fp+fs))

# energy bins centers
ph = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
Ec = 0.5*(ph[1:]+ph[:-1])
Eflu_p = (sp_p*Ec).sum(); Eflu_s = (sp_s*Ec).sum()
print('energy fluence prim/scat (MeV):', Eflu_p, Eflu_s)
print('scattered fraction energy fluence:', Eflu_s/(Eflu_p+Eflu_s))
print('mean photon energy all:', (Eflu_p+Eflu_s)/(fp+fs))
print('mean primary energy:', Eflu_p/fp, 'mean scattered:', Eflu_s/fs)

ang_p = Hp[:,:,0].sum(axis=0)/Hcnt[0]
ang_s = Hp[:,:,1].sum(axis=0)/Hcnt[1]
print('angular primary:', np.round(ang_p,6))
print('angular scattered:', np.round(ang_s,6))
np.savez('l1_data.npz', sp_p=sp_p, sp_s=sp_s, fp=fp, fs=fs, Eflu_p=Eflu_p, Eflu_s=Eflu_s, ang_p=ang_p, ang_s=ang_s)
