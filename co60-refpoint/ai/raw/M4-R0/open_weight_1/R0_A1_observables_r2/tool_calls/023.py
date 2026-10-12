import sys, os
sys.path.insert(0, os.getcwd())
src = open('mc_photon.py').read()
old = src[src.index('def seg_in_cyl_vec'):src.index('def run_mc')]
new = '''def seg_in_cyl_vec(p0,p1):
    d=p1-p0; dx,dy,dz=d[:,0],d[:,1],d[:,2]
    x0,y0,z0=p0[:,0],p0[:,1],p0[:,2]
    a=dx*dx+dy*dy; b=2*(x0*dx+y0*dy); c=x0*x0+y0*y0-CYLR*CYLR
    t0=np.zeros(len(d)); t1=np.ones(len(d))
    disc = b*b-4*a*c
    has_xy = a>1e-14
    no_hit = has_xy & (disc<=0)
    t1[no_hit]=0
    mask = has_xy & (disc>0)
    sq=np.sqrt(np.maximum(disc,0)[mask])
    ta=(-b[mask]-sq)/(2*a[mask]); tb=(-b[mask]+sq)/(2*a[mask])
    t0[mask]=np.maximum(t0[mask],np.minimum(ta,tb)); t1[mask]=np.minimum(t1[mask],np.maximum(ta,tb))
    outside = (~has_xy) & (c>0); t1[outside]=0
    az=np.abs(dz)>1e-12
    dzz=np.where(az,dz,1.0)
    tza=np.where(az,(CYLZ0-z0)/dzz,0); tzb=np.where(az,(CYLZ1-z0)/dzz,0)
    t0=np.maximum(t0,np.minimum(tza,tzb)); t1=np.minimum(t1,np.maximum(tza,tzb))
    zin = az | ((z0>=CYLZ0)&(z0<=CYLZ1)); t1[~zin]=0
    return np.maximum(t1-t0,0)*np.sqrt(dx*dx+dy*dy+dz*dz)

'''
src2 = src.replace(old, new)
open('mc_photon.py','w').write(src2)
import importlib, mc_photon
importlib.reload(mc_photon)
rng=np.random.default_rng(3); N=200000
p0=rng.uniform([-15,-15,0],[15,15,30],(N,3))
th=rng.uniform(0,2*np.pi,N); ph=np.arccos(rng.uniform(-1,1,N))
d=np.stack([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)],axis=1)
p1=p0+d*rng.exponential(15.0,N)[:,None]
L=mc_photon.seg_in_cyl_vec(p0,p1)
print('random cube hits', (L>0).sum(), 'of', N, 'frac', (L>0).mean(), 'meanL', L[L>0].mean() if (L>0).any() else 0)
print('onaxis', mc_photon.seg_in_cyl_vec(np.array([[0,0,4.95]]),np.array([[0,0,5.05]])))