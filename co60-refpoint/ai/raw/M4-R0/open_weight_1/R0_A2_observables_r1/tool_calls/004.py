import numpy as np, math, time

mc2 = 0.5109989461
RHO = 1.0

# ---------------- data ----------------
d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x = d[:,0]; mu_co = d[:,1]; mu_in = d[:,2]; mu_pe = d[:,3]; mu_pn = d[:,4]; mu_pef = d[:,5]; mu_tot = d[:,6]
s = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee = s[:,0]; Scol = s[:,1]; Srad = s[:,2]; Stot = s[:,3]; Rcsda = s[:,4]; Y = s[:,5]

def interp1(x, xx, yy):
    return np.interp(x, xx, yy)

def mu_total(E): return interp1(E, E_x, mu_tot)*RHO
def mu_coh(E):   return interp1(E, E_x, mu_co)*RHO
def mu_inc(E):   return interp1(E, E_x, mu_in)*RHO
def mu_pe_(E):   return interp1(E, E_x, mu_pe)*RHO
def mu_pair(E):  return (interp1(E, E_x, mu_pn)+interp1(E, E_x, mu_pef))*RHO

# ---------------- geometry ----------------
V_cyl = math.pi*1.0**2*0.2
ZMINC, ZMAXC = 4.9, 5.1
RC = 1.0
PHI0 = 1.0/10.5**2   # cm^-2 per photon (per line; same)

def cyl_track_vec(P0, D, L, zmin, zmax, rmax=1.0):
    # P0 (n,3), D (n,3) unit, L (n,): track length inside cylinder r<=rmax, z in [zmin,zmax]
    n = P0.shape[0]
    x0,y0,z0 = P0[:,0],P0[:,1],P0[:,2]
    dx,dy,dz = D[:,0],D[:,1],D[:,2]
    a = dx*dx+dy*dy
    b = 2*(x0*dx + y0*dy)
    c = x0*x0+y0*y0 - rmax*rmax
    t_in = np.zeros(n); t_out = L.copy()
    par = a < 1e-14
    npar = ~par
    # non-parallel
    idx = np.where(npar)[0]
    if idx.size>0:
        aa=a[idx]; bb=b[idx]; cc=c[idx]; LL=L[idx]
        disc = bb*bb - 4*aa*cc
        ok = disc >= 0
        ii = idx[ok]
        if ii.size>0:
            sq = np.sqrt(disc[ok])
            t1 = (-bb[ok]-sq)/(2*aa[ok]); t2 = (-bb[ok]+sq)/(2*aa[ok])
            tl = np.minimum(t1,t2); tu = np.maximum(t1,t2)
            t_in[ii] = np.maximum(0.0, tl)
            t_out[ii] = np.minimum(LL[ok], tu)
        # disc<0: no radial crossing; t_in stays 0, t_out stays L => empty if cannot clip? need empty:
        # for disc<0 with c>0 ray outside always; t_in/t_out default gives t_in=0,t_out=L -> wrong (would clip z only)
        # set those to t_in=t_out=0
        ii0 = idx[~ok]
        t_in[ii0]=0; t_out[ii0]=0
    # parallel
    ip = np.where(par)[0]
    if ip.size>0:
        inside0 = c[ip] <= 0
        t_in[ip[inside0]]=0; t_out[ip[inside0]]=L[ip[inside0]]
        t_in[ip[~inside0]]=0; t_out[ip[~inside0]]=0
    # z window
    adz = np.abs(dz)
    small = adz < 1e-12
    loz = np.zeros(n); hiz = L.copy()
    # non-small dz
    iz = np.where(~small)[0]
    if iz.size>0:
        z0i=z0[iz]; dzi=dz[iz]; Li=L[iz]
        t1=(zmin-z0i)/dzi; t2=(zmax-z0i)/dzi
        tl=np.minimum(t1,t2); tu=np.maximum(t1,t2)
        loz[iz]=np.maximum(0,tl); hiz[iz]=np.minimum(Li,tu)
    # small dz
    ism = np.where(small)[0]
    if ism.size>0:
        inw = (z0[ism]>=zmin)&(z0[ism]<=zmax)
        # keep defaults 0..L for in range; others zero out
        t_out[ism[~inw]]=0
    lo = np.maximum(t_in, loz)
    hi = np.minimum(t_out, hiz)
    return np.maximum(0.0, hi-lo)

# quick test
p0 = np.array([[0.,0.,4.8],[0,0,4.8],[5,0,4.8],[0,100,4.8]])
d0 = np.array([[0.,0.,1.],[0.,0.,1.],[0.,0.,1.],[0.,-1.,0.]])
L0 = np.array([0.4,10.,10.,10.])
print('cyl track test:', cyl_track_vec(p0,d0,L0,4.9,5.1,1.0))

# ---------------- bins ----------------
ph_edges = np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges = np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
cos_edges = np.linspace(-1,1,11)
NB_PH = len(ph_edges)-1; NB_EL = len(el_edges)-1; NB_COS = len(cos_edges)-1
print('bins', NB_PH, NB_EL, NB_COS)

def ph_bin(E): return np.searchsorted(ph_edges, E, side='right')-1
def el_bin(E):
    b = np.searchsorted(el_edges, E, side='right')-1
    return np.clip(b, 0, NB_EL-1)
def cos_bin(c): return np.searchsorted(cos_edges, c, side='right')-1

# KN sampling (vectorized rejection)
def sample_compton(E, rng):
    # E: array of photon energies (MeV); returns cos_s, Epr, cose (electron), acceptance loop
    n = E.size
    cos_s = np.zeros(n); Epr = np.zeros(n); cose = np.zeros(n); done = np.zeros(n, bool)
    idx = np.arange(n)
    while True:
        sub = idx[~done]
        if sub.size==0: break
        alpha = E[sub]/mc2
        cs = rng.uniform(-1,1,sub.size)
        f = KN_f(alpha, cs)
        acc = rng.random(sub.size) < (f/2.0)
        if acc.any():
            ai = sub[acc]
            cos_s[ai]=cs[acc]
            Ep = E[ai]/(1+alpha[acc]*(1-cs[acc]))
            Epr[ai]=Ep
            cse = (E[ai]-Ep*cs[acc])/np.sqrt(E[ai]**2+Ep**2-2*E[ai]*Ep*cs[acc])
            cose[ai]=cse
            done[ai]=True
    return cos_s, Epr, cose

def KN_f(alpha, cos):
    x = 1+alpha*(1-cos)
    r = 1.0/x
    return r*r*(r + x - (1-cos*cos))

def rotate_dir(d, cos, phi, rng):
    # d (n,3) unit; scatter polar angle theta with cos; returns new dirs
    sin = np.sqrt(np.maximum(0,1-cos*cos))
    sin_phi = np.sin(phi); cos_phi = np.cos(phi)
    kx,ky,kz = d[:,0],d[:,1],d[:,2]
    # e1 perpendicular to k; use cross with z unless k near z
    nearz = np.abs(kz) > 0.999
    # e1 = normalize(cross(k, ez)) = (-ky, kx, 0)/sqrt(kx^2+ky^2)
    nrm = np.sqrt(kx*kx+ky*ky)
    e1x = np.where(nearz, np.ones_like(kx), -ky/np.maximum(nrm,1e-12))
    e1y = np.where(nearz, np.zeros_like(ky), kx/np.maximum(nrm,1e-12))
    e1z = np.zeros_like(kz)
    # for near z, use e1 = xhat
    e1x[nearz]=1; e1y[nearz]=0; e1z[nearz]=0
    # e2 = cross(k, e1)
    e2x = ky*e1z - kz*e1y
    e2y = kz*e1x - kx*e1z
    e2z = kx*e1y - ky*e1x
    newx = cos*d[:,0] + sin*(cos_phi*e1x + sin_phi*e2x)
    newy = cos*d[:,1] + sin*(cos_phi*e1y + sin_phi*e2y)
    newz = cos*d[:,2] + sin*(cos_phi*e1z + sin_phi*e2z)
    out = np.stack([newx,newy,newz],axis=1)
    nrm2 = np.sqrt((out**2).sum(1))
    return out/nrm2[:,None]