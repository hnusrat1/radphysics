
code3 = r'''
import numpy as np
def cyl_track(r,d,l,zc,R=1.0,h=0.1,xc=0.0,yc=0.0):
    # track length of segments (r + s d, s in [0,l]) inside cylinder axis z, radius R, centre z=zc, half thickness h
    x0=r[:,0]-xc; y0=r[:,1]-yc; z0=r[:,2]
    dx,dy,dz=d[:,0],d[:,1],d[:,2]
    lo=np.zeros(len(l)); hi=l.copy()
    # z slab
    zlo=zc-h; zhi=zc+h
    with np.errstate(divide='ignore', invalid='ignore'):
        sa=np.where(dz!=0,(zlo-z0)/dz,np.where((z0>=zlo)&(z0<=zhi),-np.inf,np.inf))
        sb=np.where(dz!=0,(zhi-z0)/dz,np.where((z0>=zlo)&(z0<=zhi),np.inf,-np.inf))
    s1=np.minimum(sa,sb); s2=np.maximum(sa,sb)
    lo=np.maximum(lo,s1); hi=np.minimum(hi,s2)
    # radial
    A=dx*dx+dy*dy; B=2*(x0*dx+y0*dy); C=x0*x0+y0*y0-R*R
    disc=B*B-4*A*C
    with np.errstate(divide='ignore', invalid='ignore'):
        sq=np.sqrt(np.where(disc>0,disc,0))
        r1=(-B-sq)/(2*A); r2=(-B+sq)/(2*A)
    inside_rad = np.where(A>1e-14, (disc>=0), C<=0)
    r1=np.where(A>1e-14, r1, -np.inf); r2=np.where(A>1e-14, r2, np.inf)
    r1=np.where(inside_rad,r1,np.inf); r2=np.where(inside_rad,r2,-np.inf)
    lo=np.maximum(lo,r1); hi=np.minimum(hi,r2)
    return np.clip(hi-lo,0,None)
'''
open('/tmp/work/track.py','w').write(code3)
import track; importlib.reload(track)
import numpy as np
Om=0.009975072683099257; S_phi0=Om*110.0**2; V=np.pi*1.0**2*0.2
mu_p=phys.mu_all(np.array([2.0]))[4][0]; Phi_p=np.exp(-mu_p*10.0)
print("Phi_p/Phi0", Phi_p, "S_phi0", S_phi0)
t=time.time()
res=photon.run_photons(400000, seed=2)
print("time",time.time()-t, len(res['E']))
lin=track.cyl_track(res['r'],res['d'],res['l'],10.0)
sel=lin>0
Fsc=(lin*res['f']).sum()/res['N']*S_phi0/V
print("scattered fluence/Phi0 (contains only segments with f=1)",Fsc, "nseg hits", sel.sum(), "hits scattered", (sel&res['f']).sum())
