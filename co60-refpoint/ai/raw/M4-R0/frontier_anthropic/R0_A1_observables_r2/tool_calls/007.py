
import time, sys
sys.path.insert(0,'.')
import mc2, numpy as np
N=1000000
t=time.time()
depths=[0.25,0.5,0.75,1.0,2.0,3.0,5.0,7.0,10.0,15.0]
res=mc2.run(N,depths,seed=3)
print("time",time.time()-t)
tal=res['tal']
V=mc2.Vcyl; P0=mc2.Phi0_hist
for c,dz in enumerate(depths):
    Fp=tal['Lp'][c]/(N*V*P0); Fs=tal['Ls'][c]/(N*V*P0)
    Kc=(tal['KC_p'][c]+tal['KC_s'][c]+tal['KPE_p'][c]+tal['KPE_s'][c]+tal['KPP_p'][c]+tal['KPP_s'][c])/(N*V*P0)
    Kc_s=(tal['KC_s'][c]+tal['KPE_s'][c]+tal['KPP_s'][c])/(N*V*P0)
    print(f"d={dz:5.2f} Fp={Fp:.4f} Fs={Fs:.4f} s/p={Fs/Fp:.3f} Kcol(MeV cm2/g)={Kc:.5f} scatdose={Kc_s/Kc:.3f}")
