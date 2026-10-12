
import sys, time, numpy as np
sys.path.insert(0,'/tmp/work')
import phys, mc
for N in [100000,400000]:
    vals=[]
    for s in range(4):
        r=mc.run(N,[0.5,5.0],np.random.default_rng(10+s),store_for=5.0)
        vals.append(r['dscat']/N)
    vals=np.array(vals)
    print(N, vals.mean(0), vals.std(0)/np.sqrt(len(vals)))
