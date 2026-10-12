
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, mcph, mcph2, xs
rng=np.random.default_rng(5)
# simple check: attenuation of straight primaries to z=9.9 with mt
mt=xs.mut(2.0); print(mt)
s=-np.log(rng.uniform(size=200000))/mt
print((s>10/np.cos(0)).mean(), np.exp(-mt*10))
r=mcph2.run(200000,2)
print(r['flu_E'][0].sum()/200000, 0.00523*np.exp(-mt*10))
