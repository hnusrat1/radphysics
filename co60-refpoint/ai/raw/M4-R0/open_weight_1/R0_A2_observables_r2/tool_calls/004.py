import numpy as np, time

def load_xcom():
    with open('/tmp/data/xcom_water.txt') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    x = np.loadtxt(lines)
    return x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6]
E_x, mu_coh, mu_incoh, mu_pe, mu_pairn, mu_paire, mu_tot = load_xcom()
def mu_interp(E, col):
    return np.exp(np.interp(np.log(E), np.log(E_x), np.log(col)))
def mu_total(E): return mu_interp(E, mu_tot)
def mu_incoh_E(E): return mu_interp(E, mu_incoh)
def mu_coh_E(E): return mu_interp(E, mu_coh)
def mu_pe_E(E): return mu_interp(E, mu_pe)
def mu_pair_E(E): return mu_interp(E, mu_pairn+mu_paire)

mc2=0.51099895; re2=7.9408e-26; rho=1.0
def kn_dsig(E0, cosT):
    a = E0/mc2
    Ep = E0/(1.0 + a*(1.0 - cosT))
    r = Ep/E0
    return 0.5*re2 * r**2 * (r + 1.0/r - (1.0-cosT**2))
def kn_total(E0):
    th = np.linspace(1e-6, np.pi, 20000)
    return np.trapezoid(kn_dsig(E0, np.cos(th))*2*np.pi*np.sin(th), th)
def compton_Eprime(E0, cosT):
    a = E0/mc2
    return E0/(1.0 + a*(1.0 - cosT))

# photon energy bins (28) and cos theta bins (10)
pe = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4]
cte = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
pe = np.array(pe); cte=np.array(cte)

# fine energy grid for scattered fluence
fineE = np.concatenate([np.array([0.01]), np.linspace(0.02,1.4,400)])
fineE_edges = np.concatenate([np.array([0.005]), np.linspace(0.015,1.4,401)])  # not necessary, use centers and bins

def single_scatter_line(zP, E0, nz=120, nrho=60, nphi=48):
    """Return arrays over scatter points: Ep (scattered energy), cosP (direction at P),
       w = dPhi_sc/Phi0_line contribution (per point, not per dV yet? we include dV)."""
    z = np.linspace(0.02, 29.98, nz)
    rho = None
    # rho max = 5*(1+z/100) but keep within phantom lateral 15 cm (field at 30 cm depth half-width 6.5 cm < 15)
    rhomax = 5.0*(1.0 + z/100.0)
    # build grid
    Z, R, PHI = [], [], []
    for zi, rmax in zip(z, rhomax):
        rr = np.linspace(0.0, rmax*0.999, nrho)
        ph = np.linspace(0, 2*np.pi, nphi, endpoint=False)
        Rr, Ph = np.meshgrid(rr, ph, indexing='ij')
        Z.append(np.full((nrho,nphi), zi)); R.append(Rr); PHI.append(Ph)
    Z = np.concatenate(Z).ravel(); R = np.concatenate(R).ravel(); PHI = np.concatenate(PHI).ravel()
    N = Z.size
    cosphi = np.cos(PHI); sinphi = np.sin(PHI)
    # source at (0,0,-100), P on axis at zP
    d_src = np.sqrt((100.0+Z)**2 + R**2)
    rhos = R*100.0/(100.0+Z)
    d_entry = np.sqrt(100.0**2 + rhos**2)
    d_in = d_src - d_entry
    # primary fluence ratio at x (per Phi0_line)
    att_in = np.exp(-mu_total(E0)*d_in)
    phip = (105.0/d_src)**2 * att_in
    # direction to P
    r_sp = np.sqrt(R**2 + (Z-zP)**2)
    u_out_z = (zP - Z)/r_sp
    # u_in dot u_out
    uin_dot_uout = (-R*cosphi*R*0.0 - R*cosphi*0.0 + 0)  # placeholder
    # compute dot: u_in = (R cosphi, R sinphi, 100+Z)/d_src ; u_out = (-R cosphi, -R sinphi, zP-Z)/r_sp
    uin_x = R*cosphi/d_src; uin_y = R*sinphi/d_src; uin_z = (100.0+Z)/d_src
    uout_x = -R*cosphi/r_sp; uout_y = -R*sinphi/r_sp; uout_z = u_out_z
    cosT_sc = uin_x*uout_x + uin_y*uout_y + uin_z*uout_z
    # scattered photon energy
    Ep = compton_Eprime(E0, cosT_sc)
    # KN angular probability P(cosT) dOmega -> P(cosT)/r_sp^2 etc: P(cosT) per sr = dsigma/dOmega / sigma
    Pkn = kn_dsig(E0, cosT_sc)/kn_total(E0)
    mu_c = mu_incoh_E(E0)  # cm^-1 (rho=1)
    att_out = np.exp(-mu_total(Ep)*r_sp)
    # weight per unit volume (cm^-3) per Phi0_line: phip * mu_c * Pkn * exp(-mu r)/r^2
    w = phip * mu_c * Pkn * att_out / r_sp**2
    # dV = R dR dphi dz; approximate with cell volumes later; here return per-volume weights and coordinates
    return Z, R, phip, w, Ep, uout_z, d_in, r_sp

# We'll integrate using quadrature over the grid: dV = R dR dphi dz. We use simple midpoints.
# Build grid with midpoints for quadrature: z bin widths, rho bin widths, phi bin width.
def build_grid(zP, E0, nz=120, nrho=60, nphi=48):
    z = np.linspace(0.02, 29.98, nz)
    dz = np.gradient(z)
    rhomax = 5.0*(1.0 + z/100.0)
    Zall,Rall,Wall,Zw,Rw,PHIw = [],[],[],[],[],[]
    dphi = 2*np.pi/nphi
    for zi, rmax, dzi in zip(z, rhomax, dz):
        rr = np.linspace(0.0, rmax*0.999, nrho)
        dr = np.gradient(rr)
        ph = np.linspace(0, 2*np.pi, nphi, endpoint=False)
        Rr, Ph = np.meshgrid(rr, ph, indexing='ij')
        Dr = np.broadcast_to(dr[:,None], Rr.shape)
        dV = Rr*Dr* dzi * dphi
        Zall.append(np.full((nrho,nphi), zi)); Rall.append(Rr); Wall.append(dV); Zw.append(np.full((nrho,nphi),zi)); Rw.append(Rr); PHIw.append(Ph)
    return (np.concatenate(Zall).ravel(), np.concatenate(Rall).ravel(), np.concatenate(Wall).ravel(),
            np.concatenate(Zw).ravel(), np.concatenate(Rw).ravel(), np.concatenate(PHIw).ravel())

def single_scatter_total(zP, lines=(1.1732,1.3325), nz=120, nrho=60, nphi=48):
    """Returns arrays for the combined spectrum of scattered photon fluence at P (per Phi0_total):
       Ep_arr, cosP_arr, w_arr (per unit Phi0), for all scatter points and both lines."""
    Ep_all=[]; cosP_all=[]; w_all=[]
    for E0 in lines:
        Z,R,dV,Zg,Rg,PHIw = build_grid(zP, E0, nz, nrho, nphi)
        d_src = np.sqrt((100.0+Zg)**2 + Rg**2)
        rhos = Rg*100.0/(100.0+Zg)
        d_in = d_src - np.sqrt(100.0**2+rhos**2)
        phip = (105.0/d_src)**2 * np.exp(-mu_total(E0)*d_in)
        r_sp = np.sqrt(Rg**2 + (Zg-zP)**2)
        uin_x = Rg*np.cos(PHIw)/d_src; uin_y = Rg*np.sin(PHIw)/d_src; uin_z = (100.0+Zg)/d_src
        uout_x = -Rg*np.cos(PHIw)/r_sp; uout_y = -Rg*np.sin(PHIw)/r_sp; uout_z = (zP-Zg)/r_sp
        cosT_sc = uin_x*uout_x + uin_y*uout_y + uin_z*uout_z
        Ep = compton_Eprime(E0, cosT_sc)
        Pkn = kn_dsig(E0, cosT_sc)/kn_total(E0)
        att_out = np.exp(-mu_total(Ep)*r_sp)
        w = phip * mu_incoh_E(E0) * Pkn * att_out / r_sp**2 * dV
        Ep_all.append(Ep); cosP_all.append(uout_z); w_all.append(w)
    Ep_all = np.concatenate(Ep_all); cosP_all=np.concatenate(cosP_all); w_all=np.concatenate(w_all)
    return Ep_all, cosP_all, w_all

t0=time.time()
zP=5.0
Ep, cosP, w = single_scatter_total(zP)
print("time", time.time()-t0, "npoints", Ep.size)
print("total single-scatter fluence per Phi0:", w.sum())
# spectrum in the 28 bins
hist, _ = np.histogram(Ep, bins=pe, weights=w)
print("sum spectrum bins:", hist.sum())
print("angular:", np.histogram(cosP, bins=cte, weights=w)[0])
print("mean Ep of scatter:", np.average(Ep, weights=w))