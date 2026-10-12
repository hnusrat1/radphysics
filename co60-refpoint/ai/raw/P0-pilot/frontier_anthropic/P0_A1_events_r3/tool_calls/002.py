
rng=np.random.default_rng(12345)
ZE=8  # water
def mu_comp(E): return ne*sigKN(E)
def mu_pair(E): return np.where(E>2*me, 0.00048*np.clip((E-2*me)/(2-2*me),0,None)**1.0*0+0.00048*(E>2*me)*np.minimum(1,(E-2*me)/(2-2*me)),0.0)
def mu_coh(E): return 0.0051*(0.1/E)**2
def mu_pe(E): return 2.76e-3*(0.1/E)**3
def mu_tot_parts(E):
    return mu_comp(E), mu_pair(E), mu_coh(E), mu_pe(E)

def sample_KN(E,n):
    # rejection on mu uniform in [-1,1]
    k=E/me
    out=np.empty(n); 
    todo=np.arange(n)
    mu_grid=np.linspace(-1,1,20001)
    eps=1/(1+k*(1-mu_grid)); f=eps**2*(eps+1/eps-(1-mu_grid**2))
    fmax=f.max()*1.02
    while todo.size:
        m=rng.uniform(-1,1,todo.size)
        e=1/(1+k*(1-m))
        f=e**2*(e+1/e-(1-m**2))
        acc=rng.uniform(0,fmax,todo.size)<f
        out[todo[acc]]=m[acc]
        todo=todo[~acc]
    return out

def sample_rayleigh_mu(n):
    out=np.empty(n); todo=np.arange(n)
    while todo.size:
        m=rng.uniform(-1,1,todo.size)
        acc=rng.uniform(0,1,todo.size)<(1+m**2)/2
        out[todo[acc]]=m[acc]; todo=todo[~acc]
    return out

def rotate(d, mu, phi):
    # d: (n,3) unit; returns new directions with cos=mu relative to d and azimuth phi
    n=d.shape[0]
    a=np.where(np.abs(d[:,0:1])<0.9, np.array([[1.,0,0]]), np.array([[0,1.,0]]))
    e1=np.cross(d,a); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(d,e1)
    s=np.sqrt(np.clip(1-mu**2,0,None))
    return d*mu[:,None]+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)

def iso_dirs(n):
    c=rng.uniform(-1,1,n); p=rng.uniform(0,2*np.pi,n); s=np.sqrt(1-c*c)
    return np.stack([s*np.cos(p),s*np.sin(p),c],1)

def angle_dir_from_axis(d, theta):
    # electron direction with polar angle theta about d, random azimuth
    phi=rng.uniform(0,2*np.pi,d.shape[0])
    return rotate(d,np.cos(theta),phi)

# geometry
L_src=100.0; HALF=5.0
LO=np.array([-15.,-15.,0.]); HI=np.array([15.,15.,30.])
def start_photons(N):
    # uniform in solid angle within pyramid: sample plane point uniformly, accept with cos^3
    pts=[]; cnt=0
    xs=[]
    while cnt<N:
        m=int(2*N)
        x=rng.uniform(-HALF,HALF,m); y=rng.uniform(-HALF,HALF,m)
        r=np.sqrt(L_src**2+x*x+y*y)
        acc=rng.uniform(0,1,m)< (L_src/r)**3
        xs.append(np.stack([x[acc],y[acc],np.full(acc.sum(),L_src)],1)); cnt+=acc.sum()
    v=np.concatenate(xs)[:N]
    d=v/np.linalg.norm(v,axis=1)[:,None]
    # source at (0,0,-100): plane hits at z=0 after t=100/uz
    src=np.array([0,0,-L_src])
    t=(0-src[2])/d[:,2]
    p=src+t[:,None]*d
    return p,d
print("ok")
