import numpy as np
from mc_module import run_photon_mc

for label,kw in [("10x10,30deep",dict(N=200000,field_half=5.0,HW=15,ZD=30,seed=11)),
                 ("2x2,30deep",dict(N=200000,field_half=1.0,HW=15,ZD=30,seed=12)),
                 ("10x10,10deep",dict(N=200000,field_half=5.0,HW=15,ZD=10,seed=13)),
                 ("10x10,40x40lat",dict(N=200000,field_half=5.0,HW=20,ZD=30,seed=14))]:
    res=run_photon_mc(**kw)
    pt=res['prim_tl'][0]; st=res['scat_tl'][0]
    print(f"{label}: time={res['time']:.1f}s iters={res['iters']}")
    print(f"   scat/prim track = {st/pt:.4f}   eflu_scat/eflu_prim={res['eflu_scat']/max(res['eflu_prim'],1e-12):.4f}")
    print(f"   prim_tl/hist={pt/res['N']:.5f}  scat_tl/hist={st/res['N']:.5f}")
    print(f"   mean scat E = {res['eflu_scat']/max(st,1e-12):.3f} MeV")
    print(f"   int_main={res['int_main']}")
    print()