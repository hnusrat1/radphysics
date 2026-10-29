import numpy as np, time
import mcbeam as mb
t0=time.time()
r=mb.photon_mc(200000, rng=np.random.default_rng(42))
print('time',time.time()-t0,'niter',r['niter'])
print('L0_sum',r['L0_sum'])
print('ph_p_tot',r['ph_p_tot'],'ph_s_tot',r['ph_s_tot'])
print('prim ratio',r['ph_p_tot']/r['L0_sum'], 'total ratio',(r['ph_p_tot']+r['ph_s_tot'])/r['L0_sum'])
print('L2p',r['L2p'],'L2s',r['L2s'])
print('nL3',r['nL3'],'Kdep',r['Kdep'],'sumE_all',r['sumE_all'])
src=r.get('src')
if src is not None:
    print('src n',len(src['E']),'Emean',src['E'].mean(),'zmean',src['Z'].mean(),'frac_pos',src['pos'].mean())
    print('prim frac among src', src['prim'].mean())
print('spec_p sum',r['spec_p'].sum(),'spec_s sum',r['spec_s'].sum())
print('ang_p sum',r['ang_p'].sum(),'ang_s sum',r['ang_s'].sum())