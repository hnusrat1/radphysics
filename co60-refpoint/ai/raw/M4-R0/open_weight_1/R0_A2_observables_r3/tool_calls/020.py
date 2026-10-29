import importlib.util, os, numpy as np, time, json
spec=importlib.util.spec_from_file_location("mc_module", os.path.join(os.getcwd(),"mc_module.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

depths=np.array([0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0,15.0,20.0,25.0,29.0])
t0=time.time()
res=mod.run_photon_mc(N=2000000, depths=depths, seed=20240607, HW=15, ZD=30, maxiter=80)
print("MC time:", time.time()-t0, "iters", res['iters'])
np.savez('mc_results.npz', depths=depths, prim_tl=res['prim_tl'], scat_tl=res['scat_tl'],
         spec_prim=res['spec_prim'], spec_scat=res['spec_scat'], ang_prim=res['ang_prim'],
         ang_scat=res['ang_scat'], eflu_prim=res['eflu_prim'], eflu_scat=res['eflu_scat'],
         int_main=res['int_main'], N=res['N'])
print("saved. int_main:", res['int_main'])
print("scat/prim track at 5:", res['scat_tl'][np.argmin(abs(depths-5))]/res['prim_tl'][np.argmin(abs(depths-5))])