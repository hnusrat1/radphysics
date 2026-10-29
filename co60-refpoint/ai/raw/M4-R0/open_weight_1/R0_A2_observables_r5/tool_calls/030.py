import sys, time, numpy as np
sys.path.insert(0,'/tmp')
import electronmc, importlib
importlib.reload(electronmc)
d = np.load('/tmp/photon_sources.npz')
src_pos=d['pos'][:200000]; src_dir=d['dir'][:200000]; src_T=d['T'][:200000]
src_par=d['par'][:200000]; src_kind=d['kind'][:200000]
depths=np.array([0.2,0.35,0.5,0.65,0.8,1.0,1.5,2,3,4,5,6,8,10])
t0=time.time()
res=electronmc.electron_run(src_pos, src_dir, src_T, src_par, src_kind, np.ones(len(src_T)), depths, seed=3, verbose=True)
print('elapsed', time.time()-t0)
print('dose', res['dose'])
print('dose_scat', res['dose_scat'])
print('flu_tot5', res['flu_tot5'], 'flu_delta5', res['flu_delta5'])
print('spectrum', res['flu_e_spectrum']/res['flu_tot5'] if res['flu_tot5']>0 else None) 
print('angular', res['flu_angular']/res['flu_tot5'])
print('rad/col', res['rad_col_num']/res['rad_col_den'])