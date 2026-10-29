
import numpy as np
def ray_cylinder_hit(p, u, z0=5.0, R=1.0, h=0.1):
    N=len(p)
    a = u[:,0]**2+u[:,1]**2
    b = 2*(p[:,0]*u[:,0]+p[:,1]*u[:,1])
    c = p[:,0]**2+p[:,1]**2-R**2
    disc=b*b-4*a*c
    t_in=np.full(N,np.inf); t_out=np.full(N,-np.inf)
    m=a>1e-12
    sd=np.sqrt(np.maximum(disc,0))
    t1=np.where(m,(-b-sd)/(2*a),np.inf); t2=np.where(m,(-b+sd)/(2*a),np.inf)
    for t in [t1,t2]:
        z=p[:,2]+t*u[:,2]
        ok=(z>=z0-h)&(z<=z0+h)&(t>=-1e-7)
        tv=np.maximum(t,0)
        t_in=np.minimum(t_in,np.where(ok,tv,np.inf))
        t_out=np.maximum(t_out,np.where(ok,tv,-np.inf))
    for zc in [z0-h,z0+h]:
        tz=(zc-p[:,2])/u[:,2]
        xz=p[:,0]+tz*u[:,0]; yz=p[:,1]+tz*u[:,1]
        ok=(xz**2+yz**2<=R*R+1e-10)&(tz>=-1e-7)
        t_in=np.minimum(t_in,np.where(ok,np.maximum(tz,0),np.inf))
        t_out=np.maximum(t_out,np.where(ok,np.maximum(tz,0),-np.inf))
    hit=t_in<np.inf
    r2=p[:,0]**2+p[:,1]**2
    inside=(r2<=R*R)&(p[:,2]>=z0-h)&(p[:,2]<=z0+h)
    t_in=np.where(inside,0.0,t_in)
    return hit, t_in, t_out

rng=np.random.default_rng(0)
N=200000
p=np.zeros((N,3)); p[:,2]=3.0
u=rng.normal(size=(N,3)); u/=np.linalg.norm(u,axis=1,keepdims=True)
hit,_,_=ray_cylinder_hit(p,u)
print("on-axis z=3 isotropic hit fraction:",hit.mean()," expected ~0.0625 (end face) + side")

# off-axis z=3, r=5
p[:,0]=5.0
u=rng.normal(size=(N,3)); u/=np.linalg.norm(u,axis=1,keepdims=True)
hit,_,_=ray_cylinder_hit(p,u)
print("off-axis r=5 z=3 isotropic hit fraction:",hit.mean())

# directions forward-biased from a point r=5: hit fraction?
# sample directions with cos>0.9
th=rng.uniform(0,0.45,size=N); ph=rng.uniform(0,2*np.pi,size=N)
u=np.stack([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)],axis=1)
hit,_,_=ray_cylinder_hit(p,u)
print("off-axis r=5 z=3 tightly-forward hit fraction:",hit.mean())
