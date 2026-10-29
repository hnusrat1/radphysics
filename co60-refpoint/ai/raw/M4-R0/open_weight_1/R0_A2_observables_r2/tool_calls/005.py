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

def build_grid(zP, E0, nz=120, nrho=60, nphi=48, zmax=29.98):
    z = np.linspace(0.02, zmax, nz)
    dz = np.gradient(z)
    rhomax = 5.0*(1.0 + z/100.0)
    Zall,Rall,Wall,Zm,Rm,PHIm = [],[],[],[],[],[]
    dphi = 2*np.pi/nphi
    for zi, rmax, dzi in zip(z, rhomax, dz):
        rr = np.linspace(0.0, rmax*0.999, nrho)
        dr = np.gradient(rr)
        ph = np.linspace(0, 2*np.pi, nphi, endpoint=False)
        Rr, Ph = np.meshgrid(rr, ph, indexing='ij')
        Dr = np.broadcast_to(dr[:,None], Rr.shape)
        dV = Rr*Dr* dzi * dphi
        Zall.append(np.full((nrho,nphi), zi)); Rall.append(Rr); Wall.append(dV)
        Zm.append(np.full((nrho,nphi),zi)); Rm.append(Rr); PHIm.append(Ph)
    return (np.concatenate(Zall).ravel(), np.concatenate(Rall).ravel(), np.concatenate(Wall).ravel(),
            np.concatenate(Zm).ravel(), np.concatenate(Rm).ravel(), np.concatenate(PHIm).ravel())

def single_results(zP, E0, nz=120, nrho=60, nphi=48):
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
    return w, Ep, uout_z, Zg

for zP in [0.5, 2.5, 5.0, 10.0]:
    tot = 0
    for E0 in [1.1732,1.3325]:
        w,Ep,cosP,Zg = single_results(zP,E0)
        tot += 0.5*w.sum()
    print(f"zP={zP}: single-scatter per Phi0 total = {tot:.4f}")

# upstream/downstream split at zP=5
zP=5.0
tot=0; up=0; dn=0; near=0
for E0 in [1.1732,1.3325]:
    w,Ep,cosP,Zg = single_results(zP,E0)
    tot += 0.5*w.sum()
    up += 0.5*w[Zg<zP-0.1].sum()
    dn += 0.5*w[Zg>zP+0.1].sum()
    near += 0.5*w[(Zg>=zP-0.1)&(Zg<=zP+0.1)].sum()
print("z=5 split: total", tot, "upstream", up, "downstream", dn, "near(within 0.1cm)", near)

# convergence: increase grid
for nz,nrho,nphi in [(120,60,48),(160,80,64),(200,100,80)]:
    tot=0
    for E0 in [1.1732,1.3325]:
        w,Ep,cosP,Zg = single_results(zP,E0,nz,nrho,nphi)
        tot += 0.5*w.sum()
    print("grid",nz,nrho,nphi,"total",tot)