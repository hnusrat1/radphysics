def seg_cyl_len(p0, p1, z0=5.0, R=1.0, hz=0.2):
    x0=p0[:,0]; y0=p0[:,1]; z0p=p0[:,2]
    sx=p1[:,0]-x0; sy=p1[:,1]-y0; sz=p1[:,2]-z0p
    L=np.sqrt(sx*sx+sy*sy+sz*sz)
    N=len(L)
    dx=np.zeros(N); dy=np.zeros(N); dz=np.zeros(N)
    nz=L>1e-15
    dx[nz]=sx[nz]/L[nz]; dy[nz]=sy[nz]/L[nz]; dz[nz]=sz[nz]/L[nz]
    a = (dx**2+dy**2)*L**2
    b = 2*(x0*dx+y0*dy)*L
    c = x0**2+y0**2-R**2
    t1=np.zeros(N); t2=np.ones(N)
    par = a<1e-15   # parallel to axis: radial distance const
    # for non-parallel: quadratic
    npar=~par
    disc = b[npar]**2-4*a[npar]*c[npar]
    sq=np.sqrt(np.maximum(disc,0))
    t1[npar] = (-b[npar]-sq)/(2*a[npar])
    t2[npar] = (-b[npar]+sq)/(2*a[npar])
    # parallel case: inside radius if c<=0
    t1[par]=0.0; t2[par]=1.0
    ok = np.ones(N,bool)
    q_ok = disc>0
    ok[npar] = q_ok
    # for parallel, need c<=0; for non-par, roots real
    bad = np.zeros(N,bool)
    bad[par] = c[par]>0
    bad[npar] = ~q_ok
    tLo_r = np.clip(t1,0,1); tHi_r = np.clip(t2,0,1)
    zlo=z0-hz/2; zhi=z0+hz/2
    tz_lo = np.where(np.abs(dz)>1e-12, (zlo-z0p)/(dz*L), -1e9)
    tz_hi = np.where(np.abs(dz)>1e-12, (zhi-z0p)/(dz*L),  1e9)
    tLo_z = np.minimum(tz_lo,tz_hi); tHi_z=np.maximum(tz_lo,tz_hi)
    tLo = np.maximum(tLo_r, np.clip(tLo_z,0,1))
    tHi = np.minimum(tHi_r, np.clip(tHi_z,0,1))
    good = (tHi>tLo)&~bad
    length=np.zeros(N); length[good]=L[good]*(tHi[good]-tLo[good])
    return length, good

p0=np.array([[0,0,4.9],[1.5,0,5.0],[0.5,0,4.8],[0,0,4.95]])
p1=np.array([[0,0,5.1],[1.5,0,5.2],[1.2,0.7,5.4],[0,0,5.05]])
print(seg_cyl_len(p0,p1)[0])