
driver = r'''
import sys, json, time, gc
sys.path.insert(0,'/tmp/work')
import numpy as np, final
seeds=range(int(sys.argv[1]) if len(sys.argv)>1 else 16)
outs=[]
t=time.time()
for s in seeds:
    o,B=final.compute(100000,500+s,Nel=20000)
    outs.append(o); del B; gc.collect()
print('time',time.time()-t)
keys=[k for k in outs[0] if not k.startswith('_')]
res={}
for k in keys:
    vals=[o[k] for o in outs]
    if isinstance(vals[0],dict):
        res[k]={a:float(np.mean([v[a] for v in vals])) for a in vals[0]}
    elif isinstance(vals[0],list):
        res[k]=np.mean(np.array(vals,float),0).tolist()
    else:
        res[k]=float(np.mean(vals))
json.dump(dict(res=res, outs=outs), open('/tmp/work/results.json','w'))
for key in ['L1_scattered_fraction_fluence','L1_scattered_fraction_energy_fluence','L5_dose_fraction_parent_scattered','L5_K_over_phi0_Gy_cm2','L5_s_w_air','L4_delta_fraction','L2_scattered_share_of_interactions','L4_electron_fluence_per_phi0','L1_total_fluence_per_phi0','L4_radiative_to_collision','L5_g']:
    v=np.array([o[key] for o in outs]); print(key, 'mean',v.mean(),'sem',v.std(ddof=1)/np.sqrt(len(v)))
'''
open('/tmp/work/driver.py','w').write(driver)
import subprocess
r=subprocess.run(['python3','/tmp/work/driver.py','16'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-4000:], r.stderr[-3000:])
