import numpy as np, time

# ---------- vectorized intersection of photon segment [p0,p1] with cylinder (z0, R, hz) ----------
def seg_cyl_len(p0, p1, z0=5.0, R=1.0, hz=0.2):
    # p0,p1 arrays (N,3); returns length inside cylinder (N,), and whether crossed
    x0=p0[:,0]; y0=p0[:,1]; z0p=p0[:,2]
    sx=p1[:,0]-x0; sy=p1[:,1]-y0; sz=p1[:,2]-z0p
    L=np.sqrt(sx*sx+sy*sy+sz*sz)
    # handle zero-length (shouldn't happen)
    nz = L>0
    dx=np.zeros_like(L); dy=np.zeros_like(L); dz=np.zeros_like(L)
    dx[nz]=sx[nz]/L[nz]; dy[nz]=sy[nz]/L[nz]; dz[nz]=sz[nz]/L[nz]
    # radial quadratic on t in [0,1]
    a = (dx**2+dy**2)*L**2
    b = 2*(x0*dx+y0*dy)*L
    c = x0**2+y0**2-R**2
    disc = b*b-4*a*c
    length=np.zeros_like(L)
    ok = disc>0
    # radial interval [t1,t2]
    t1=np.zeros_like(L); t2=np.ones_like(L)
    sq=np.sqrt(np.maximum(disc,0))
    # t = (-b ± sqrt(disc))/(2a); handle a~0
    t1[ok] = (-b[ok]-sq[ok])/(2*a[ok])
    t2[ok] = (-b[ok]+sq[ok])/(2*a[ok])
    tLo_r = np.clip(t1,0,1); tHi_r = np.clip(t2,0,1)
    bad = (tHi_r < tLo_r) | ~ok
    # z interval
    zlo=z0-hz/2; zhi=z0+hz/2
    tz_lo = np.where(np.abs(dz)>1e-12, (zlo-z0p)/(dz*L), -1e9)
    tz_hi = np.where(np.abs(dz)>1e-12, (zhi-z0p)/(dz*L),  1e9)
    tLo_z = np.minimum(tz_lo,tz_hi); tHi_z=np.maximum(tz_lo,tz_hi)
    tLo = np.maximum(tLo_r, np.clip(tLo_z,0,1))
    tHi = np.minimum(tHi_r, np.clip(tHi_z,0,1))
    good = (tHi>tLo)&~bad
    length[good] = L[good]*(tHi[good]-tLo[good])
    return length, good

# quick test: primary photon straight down axis through cylinder at z=5: p0=[0,0,4.9], p1=[0,0,5.1]
p0=np.array([[0,0,4.9]]); p1=np.array([[0,0,5.1]])
print("axis test:", seg_cyl_len(p0,p1)[0])
# off-axis
p0=np.array([[1.5,0,5.0]]); p1=np.array([[1.5,0,5.2]])
print("off-axis:", seg_cyl_len(p0,p1)[0])
# oblique
p0=np.array([[0.5,0,4.8]]); p1=np.array([[1.2,0.7,5.4]])
print("oblique:", seg_cyl_len(p0,p1)[0])

# ---------- rotation helpers ----------
def rotate_dir(d, cosT, sinT, phi):
    # d: (N,3); returns rotated unit directions making angle theta with d
    # build basis e1 perp to d, e2 = d x e1
    N=d.shape[0]
    # arbitrary reference not parallel to d
    ref=np.tile(np.array([0.,0.,1.]),(N,1))
    # choose ref not parallel
    par = np.abs(d[:,2])>0.9
    ref[par]=[1.,0.,0.]
    e1 = np.cross(ref,d); e1 /= np.linalg.norm(e1,axis=1,keepdims=True)+1e-30
    e2 = np.cross(d,e1)
    cT=cosT[:,None]; sT=sinT[:,None]; cp=np.cos(phi)[:,None]; sp=np.sin(phi)[:,None]
    return cT*d + sT*(cp*e1+sp*e2)

# ---------- ionization / photoelectron dir ----------
def sample_iso_dir(N):
    z=np.random.uniform(-1,1,N); phi=np.random.uniform(0,2*np.pi,N)
    r=np.sqrt(1-z*z)
    return np.stack([r*np.cos(phi), r*np.sin(phi), z],axis=1)

def sample_forward_dir(N, spread=0.3):
    # forward peaked, cos ~ 1 - |exp| small: use mu = 1 - spread*|gauss|
    mu = np.minimum(1.0, 1.0 - spread*np.abs(np.random.normal(0,1,N)))
    phi=np.random.uniform(0,2*np.pi,N); r=np.sqrt(1-mu**2)
    return np.stack([r*np.cos(phi), r*np.sin(phi), mu],axis=1)

print("helpers ok")