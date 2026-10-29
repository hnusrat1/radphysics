import numpy as np
r=np.load('/tmp/photon_mc_final.npy',allow_pickle=True).item()
N=r['N']; Omega105=109.97517633116931; Vcyl=np.pi*1.0**2*0.2
print("N=",N,"time",r['time'],"pair",r['pair_spawn'])
f_prim_tot = r['flu_prim'].sum()/N/Vcyl*Omega105
f_sec_tot  = r['flu_sec'].sum()/N/Vcyl*Omega105
print("phi_prim/Phi0 =",f_prim_tot," phi_sec/Phi0 =",f_sec_tot)
print("scattered fluence fraction:",f_sec_tot/(f_prim_tot+f_sec_tot))
Eflu_p = r['Eflu_prim']/N/Vcyl*Omega105
Eflu_s = r['Eflu_sec']/N/Vcyl*Omega105
print("phiE_prim/Phi0 (MeV cm^-1 per Phi0?)",Eflu_p," phiE_sec",Eflu_s)
print("scattered energy fluence fraction:",Eflu_s/(Eflu_p+Eflu_s))
print("mean photon energy all:", (Eflu_p+Eflu_s)/(f_prim_tot+f_sec_tot)," primary:",Eflu_p/f_prim_tot," scattered:",Eflu_s/f_sec_tot)
# spectra
pe_edges=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
spectrum_prim = r['flu_prim']/N/Vcyl*Omega105
spectrum_sec  = r['flu_sec']/N/Vcyl*Omega105
print("\nprimary spectrum (per Phi0):")
for i in range(28):
    if spectrum_prim[i]>1e-5: print(f"  bin {pe_edges[i]:.2f}-{pe_edges[i+1]:.2f}: {spectrum_prim[i]:.5f}")
print("\nscattered spectrum (per Phi0), top bins:")
order=np.argsort(spectrum_sec)[::-1][:12]
for i in order:
    print(f"  bin {pe_edges[i]:.2f}-{pe_edges[i+1]:.2f}: {spectrum_sec[i]:.5f}")
print("sum scattered spec:",spectrum_sec.sum())
# angular
print("\nangular primary:",r['ang_prim']/r['ang_prim'].sum())
print("angular scattered:",r['ang_sec']/r['ang_sec'].sum())
# slab Kcol profile
nz=r['slab_Kcol'].shape[0]; zc=r['zc_slabs']; hhslab=r['hh_slab']; Vslab=np.pi*1*2*hhslab
Kcol_MeVg = r['slab_Kcol']/N/Vslab*Omega105
print("\nKcol(MeV/g per Phi0) vs depth:")
for k in [0,1,2,4,6,8,10,12,16,20,24,28,32]:
    if k<len(zc): print(f"  z={zc[k]:.2f}: {Kcol_MeVg[k]:.5f}  Gycm2={Kcol_MeVg[k]*1.602e-10:.3e}")
# slab primary line fluence per depth
print("\nslab primary fluence per Phi0 per line vs depth:")
for k in [0,2,4,8,12,16,20,24,28,32]:
    if k<len(zc):
        f1=r['slab_prim'][0,k]/N/Vslab*Omega105; f2=r['slab_prim'][1,k]/N/Vslab*Omega105
        print(f"  z={zc[k]:.2f}: line1={f1:.4f} line2={f2:.4f} tot={f1+f2:.4f} exp={0.5*np.exp(-0.06514178*zc[k])+0.5*np.exp(-0.06104757*zc[k]):.4f}")
np.save('/tmp/analysis1.npy',{'f_prim_tot':f_prim_tot,'f_sec_tot':f_sec_tot,'Eflu_p':Eflu_p,'Eflu_s':Eflu_s,
        'spectrum_prim':spectrum_prim,'spectrum_sec':spectrum_sec,'ang_prim':r['ang_prim']/r['ang_prim'].sum(),
        'ang_sec':r['ang_sec']/r['ang_sec'].sum(),'Kcol_MeVg':Kcol_MeVg,'zc':zc})