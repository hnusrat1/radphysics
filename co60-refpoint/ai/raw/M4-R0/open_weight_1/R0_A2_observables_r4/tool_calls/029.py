import sys, numpy as np, time
sys.path.insert(0,'/tmp')
import emc
rph=np.load('/tmp/photon_mc_final.npy',allow_pickle=True).item()
src=emc.build_source_fns(rph)
t0=time.time()
res=emc.run_electron_mc(5.0,20000,seed=101,src=src)
print("time %.1f it %d steps %d ndelta %d"%(res['time'],res['it'],res['steps'],res['n_delta']))
print("N_birth:",res['N_birth'],"w_el:",res['w_el'])
V=np.pi*0.2
print("e fluence/Phi0:",res['sumChord']/V*res['w_el'])
D=(res['sumSrest']+res['trackend'])/V*res['w_el']*1.602e-10
print("D/Phi0 Gycm2:",D)
print("dose prim frac:",res['dose_prim']/(res['dose_prim']+res['dose_sec']))
print("dose sec frac:",res['dose_sec']/(res['dose_prim']+res['dose_sec']))
print("mean E:",res['sumE']/res['sumChord'])
print("Srad/Scol:",res['sumSrad']/res['sumScol'])
print("delta flu frac:",res['flu_delta'].sum()/res['flu'].sum())
print("angular:",np.round(res['ang']/res['ang'].sum(),4))
print("spectrum:",np.round(res['flu']/res['flu'].sum(),4))
np.save('/tmp/emc_test.npy',res,allow_pickle=True)