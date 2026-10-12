
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, mcph, mcph2
rng=np.random.default_rng(0)
N=200000
m=int(N*1.1)
a=rng.uniform(-0.05,0.05,m); b=rng.uniform(-0.05,0.05,m)
ok=rng.uniform(0,1,m)<(1+a*a+b*b)**(-1.5)
ab=np.stack([a[ok],b[ok]],1)
D=np.column_stack([ab[:,0],ab[:,1],np.ones(len(ab))]); D/=np.linalg.norm(D,axis=1)[:,None]
t0=100/D[:,2]; P=np.column_stack([D[:,0]*t0,D[:,1]*t0,np.zeros(len(ab))])
L,lo,hi=mcph2.chord(P,D,np.full(len(P),50.0),1.0,9.9,10.1)
print(len(P), (L>0).mean(), L.sum()/len(ab), np.pi/121*0.2)
print(np.mean(D[:,2]), np.sqrt(np.mean(P[:,0]**2+P[:,1]**2)))
