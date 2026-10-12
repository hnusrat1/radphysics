import importlib.util, os
spec=importlib.util.spec_from_file_location("mc_module", os.path.join(os.getcwd(),"mc_module.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print("ok", mod.run_photon_mc)
import numpy as np
for label,kw in [("10x10,30deep",dict(N=150000,field_half=5.0,HW=15,ZD=30,seed=11)),
                 ("2x2,30deep",dict(N=150000,field_half=1.0,HW=15,ZD=30,seed=12)),
                 ("10x10,10deep",dict(N=150000,field_half=5.0,HW=15,ZD=10,seed=13))]:
    res=mod.run_photon_mc(**kw)
    pt=res['prim_tl'][0]; st=res['scat_tl'][0]
    print(f"{label}: time={res['time']:.1f}s")
    print(f"   scat/prim track={st/pt:.4f}  eflu_s/p={res['eflu_scat']/max(res['eflu_prim'],1e-12):.4f}")
    print(f"   mean scat E={res['eflu_scat']/max(st,1e-12):.3f} MeV")
    print(f"   int_main={res['int_main']}")
    print()