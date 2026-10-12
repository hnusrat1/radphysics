import sys, os
sys.path.insert(0, os.getcwd())
import mc_photon as mcp
import numpy as np
rng=np.random.default_rng(3)
N=20000
p0=rng.uniform([-15,-15,0],[15,15,30],(N,3))
th=rng.uniform(0,2*np.pi,N); ph=np.arccos(rng.uniform(-1,1,N))
d=np.stack([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)],axis=1)
s=rng.exponential(15.0,N)
p1=p0+d*s[:,None]
L=mcp.seg_in_cyl_vec(p0,p1)
hits=np.where(L>0)[0]
print('hits', len(hits))
for i in hits[:10]:
    print('p0',p0[i],'p1',p1[i],'L',L[i])
    # manual: solve
    dx,dy,dz=p1[i]-p0[i]; x0,y0,z0=p0[i]
    a=dx*dx+dy*dy; b=2*(x0*dx+y0*dy); c=x0*x0+y0*y0-1
    print('  a,b,c',a,b,c,'disc',b*b-4*a*c)
    if a>1e-14:
        sq=np.sqrt(b*b-4*a*c); ta=(-b-sq)/(2*a); tb=(-b+sq)/(2*a)
        print('  roots',ta,tb,'clipped',max(0,min(ta,tb)),min(1,max(ta,tb)))
    else:
        print('  a=0 c=',c)
