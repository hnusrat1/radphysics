
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, final, time, json, gc
t=time.time()
outs=[]
for s in range(8):
    o,B=final.compute(100000,100+s,Nel=20000)
    outs.append(o); del B; gc.collect()
print(time.time()-t)
def avg(key):
    vals=[o[key] for o in outs]
    if isinstance(vals[0],dict):
        return {k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
    if isinstance(vals[0],list):
        arr=np.array(vals,float); m=arr.mean(0)
        if key.startswith('L3_initial_spectrum') or key.startswith('L1_angular') or key.startswith('L4_angular') or key.startswith('L3_initial_direction') or key.startswith('L4_electron_fluence_spectrum'):
            pass
        return m.tolist()
    return float(np.mean(vals))
final_out={}
for k in outs[0]:
    if k.startswith('_'): continue
    final_out[k]=avg(k)
for k,v in final_out.items():
    if isinstance(v,list): print(k, np.round(v,5).tolist())
    elif isinstance(v,dict): print(k, {a:round(b,5) for a,b in v.items()})
    else: print(k, float(v))
print('std of L1 primary', np.std([o['L1_total_fluence_per_phi0'] for o in outs]), 'K std', np.std([o['L5_K_over_phi0_Gy_cm2'] for o in outs])/5.5e-12)
