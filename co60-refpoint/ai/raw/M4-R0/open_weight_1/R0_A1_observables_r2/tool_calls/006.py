
import numpy as np, time, json, os
from scipy.integrate import quad
from scipy.interpolate import interp1d

# ---------------- constants ----------------
r_e = 2.8179403262e-13
r_e2 = r_e*r_e
mec2 = 0.5109989461
NA = 6.02214076e23
M_w = 18.01528
n_mol = NA/M_w
n_e = 10*n_mol
TWOPI = 2*np.pi

# bins
ph_edges = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
ct_edges = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
nph = len(ph_edges)-1; nel = len(el_edges)-1; nct = len(ct_edges)-1

# ---------------- cross sections ----------------
def kn_dsig_dOm(E, costh):
    k = E/mec2
    Ep = E/(1.0 + k*(1.0-costh))
    r = Ep/E
    return 0.5*r_e2*r*r*(1.0/r + r - (1.0 - costh*costh))

def kn_total(E):
    f = lambda th: kn_dsig_dOm(E, np.cos(th))*np.sin(th)
    v,_ = quad(f, 0, np.pi, epsabs=1e-30, limit=400)
    return TWOPI*v

# coherent via independent-atom Gaussian form factor (shape), normalized by chosen mu_coh table
def ray_dsig_dOm_shape(E, costh):
    hcA = 0.01973269804
    q = E*np.sqrt(2*(1-costh))/hcA
    aO, aH = 0.25, 0.20
    FO = 8.0*np.exp(-(q*q*aO*aO)/2.0)
    FH = 1.0*np.exp(-(q*q*aH*aH)/2.0)
    F2 = FO*FO + 2.0*FH*FH
    return 0.5*r_e2*(1.0+costh*costh)*F2

# empirical mu tables (cm2/g)
def mu_photo_f(E):
    A = 0.002*(0.1)**3.3
    return A*E**(-3.3)
def mu_pair_f(E):
    if E <= 1.022: return 0.0
    return 0.0045*(E-1.022)

E_coh_pts = np.array([0.01,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.5,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0])
mu_coh_pts= np.array([0.12,0.10,0.040,0.020,0.015,0.0075,0.0048,0.0025,0.0012,0.0006,0.00045,0.00038,0.00035,0.00032,0.00028,0.00020])
def mu_coh_f(E):
    return np.interp(E, E_coh_pts, mu_coh_pts, left=0.12, right=0.00015)

def mu_comp_f(E):
    return n_e*kn_total(E)

Eg = np.linspace(0.001,1.401,1401)
mu_comp_g = np.array([mu_comp_f(e) for e in Eg])
mu_coh_g  = np.array([mu_coh_f(e) for e in Eg])
mu_pho_g  = np.array([mu_photo_f(e) for e in Eg])
mu_pri_g  = np.array([mu_pair_f(e) for e in Eg])
mu_tot_g  = mu_comp_g+mu_coh_g+mu_pho_g+mu_pri_g

def interp_from(grid, vals):
    return interp1d(grid, vals, bounds_error=False, fill_value=(vals[0], vals[-1]))
mu_comp = interp_from(Eg, mu_comp_g)
mu_coh  = interp_from(Eg, mu_coh_g)
mu_pho  = interp_from(Eg, mu_pho_g)
mu_pri  = interp_from(Eg, mu_pri_g)
mu_tot  = interp_from(Eg, mu_tot_g)

for E in [0.05,0.1,1.1732,1.3325]:
    print(E, 'mu_tot', mu_tot(E), 'comp/coh/pho/pair', mu_comp(E),mu_coh(E),mu_pho(E),mu_pri(E))

# mu_en table and interpolation
E_muen = np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0,3.0])
muen_pts= np.array([4.79,1.23,0.541,0.1537,0.0682,0.0409,0.0320,0.0260,0.0255,0.0275,0.0268,0.0288,0.0295,0.0328,0.0329,0.0322,0.0310,0.0303,0.0297,0.0294,0.0284,0.0266,0.0238])
mu_en_f = interp1d(E_muen, muen_pts, kind='cubic', bounds_error=False, fill_value='extrapolate')

# ---------------- Compton sampling tables (CDF over cos, per energy bin) ----------------
nEtab = 280  # energy table 0.001..1.4
Etab = np.linspace(0.001,1.4,nEtab)
nCtab = 2000
cos_tab = np.linspace(-1,1,nCtab+1)
cos_mid = (cos_tab[1:]+cos_tab[:-1])/2
CDFcomp = np.zeros((nEtab, nCtab+1))
for i,E in enumerate(Etab):
    dens = kn_dsig_dOm(E, cos_mid)
    cum = np.concatenate([[0], np.cumsum(dens)])
    CDFcomp[i] = cum/cum[-1]

# coherent CDF
CDFcoh = np.zeros((nEtab, nCtab+1))
for i,E in enumerate(Etab):
    dens = ray_dsig_dOm_shape(E, cos_mid)
    cum = np.concatenate([[0], np.cumsum(dens)])
    CDFcoh[i] = cum/cum[-1]

def sample_compton_cos(E, r):
    idx = np.clip(np.searchsorted(Etab, E), 0, nEtab-1)
    return np.interp(r, CDFcomp[idx], cos_tab)

def sample_coherent_cos(E, r):
    idx = np.clip(np.searchsorted(Etab, E), 0, nEtab-1)
    return np.interp(r, CDFcoh[idx], cos_tab)

# ---------------- geometry ----------------
SURF_Z = 0.0
PH = 30.0   # water cube side
CYLZ0, CYLZ1 = 4.9, 5.1
CYLR = 1.0
Vcyl = np.pi*CYLR*CYLR*(CYLZ1-CYLZ0)
Phi0_area = 10.5*10.5  # cm^2 at 105 cm

def seg_in_cyl(p0, p1):
    # p0,p1: arrays (3,) ; returns length inside cylinder
    d = p1-p0
    # xy quadratic
    dx,dy,dz = d
    x0,y0,z0 = p0
    a = dx*dx+dy*dy
    b = 2*(x0*dx+y0*dy)
    c = x0*x0+y0*y0 - CYLR*CYLR
    t0, t1 = 0.0, 1.0
    if a > 1e-14:
        disc = b*b-4*a*c
        if disc <= 0: return 0.0
        sq = np.sqrt(disc)
        ta = (-b-sq)/(2*a); tb = (-b+sq)/(2*a)
        t0 = max(t0, min(ta,tb)); t1 = min(t1, max(ta,tb))
        if t1 <= t0: return 0.0
    else:
        if c > 0: return 0.0
    if abs(dz) > 1e-12:
        tza = (CYLZ0-z0)/dz; tzb = (CYLZ1-z0)/dz
        t0 = max(t0, min(tza,tzb)); t1 = min(t1, max(tza,tzb))
    else:
        if not (CYLZ0 <= z0 <= CYLZ1): return 0.0
    if t1 <= t0: return 0.0
    return (t1-t0)*np.sqrt(dx*dx+dy*dy+dz*dz)

# also vectorized version for many segments: return length array
def seg_in_cyl_vec(p0, p1):
    d = p1-p0
    dx,dy,dz = d[:,0],d[:,1],d[:,2]
    x0,y0,z0 = p0[:,0],p0[:,1],p0[:,2]
    a = dx*dx+dy*dy
    b = 2*(x0*dx+y0*dy)
    c = x0*x0+y0*y0 - CYLR*CYLR
    t0 = np.zeros(len(d)); t1 = np.ones(len(d))
    aa = a > 1e-14
    disc = b*b-4*a*c
    ok = disc > 0
    mask = aa & ok
    sq = np.sqrt(disc[mask])
    ta = (-b[mask]-sq)/(2*a[mask]); tb = (-b[mask]+sq)/(2*a[mask])
    t0[mask] = np.maximum(t0[mask], np.minimum(ta,tb))
    t1[mask] = np.minimum(t1[mask], np.maximum(ta,tb))
    outside = (~aa) & (c > 0)
    t1[outside] = 0
    # z clip
    az = np.abs(dz) > 1e-12
    tza = np.where(az, (CYLZ0-z0)/np.where(az,dz,1), 0)
    tzb = np.where(az, (CYLZ1-z0)/np.where(az,dz,1), 0)
    lo = np.minimum(tza,tzb); hi = np.maximum(tza,tzb)
    t0 = np.maximum(t0, lo); t1 = np.minimum(t1, hi)
    zin = az | ((z0>=CYLZ0)&(z0<=CYLZ1))
    t1[~zin] = 0
    L = np.maximum(t1-t0, 0)*np.sqrt(dx*dx+dy*dy+dz*dz)
    return L

print('test seg_in_cyl', seg_in_cyl(np.array([0,0,4.95]), np.array([0,0,5.05])))

# ---------------- run MC ----------------
def run_mc(N, seed=1):
    rng = np.random.default_rng(seed)
    t0=time.time()
    # scoring arrays
    flu_pri_ph = np.zeros(nph); flu_pri_ct = np.zeros(nct)   # track length sums (cm)
    flu_sct_ph = np.zeros(nph); flu_sct_ct = np.zeros(nct)
    # interactions in cylinder
    n_int = {'comp':0,'photo':0,'pair':0,'ray':0}
    n_int_pri = {'comp':0,'photo':0,'pair':0,'ray':0}
    n_int_sct = {'comp':0,'photo':0,'pair':0,'ray':0}
    n_scat_inter = 0  # interactions of scattered photons
    E_tr_by_proc = {'comp':0.0,'photo':0.0,'pair':0.0,'ray':0.0}
    E_tr_total = 0.0; E_tr_scat = 0.0
    # electron initial: per bin (all, primary-parent), direction bins (all)
    el_ini_all = np.zeros(nel); el_ini_pri = np.zeros(nel); el_dir = np.zeros(nct)
    n_el_all=0; n_el_pri=0
    # primary/scattered photon interaction counts total (anywhere) for process fractions at point already in cylinder
    # also total interactions in whole phantom (for reference)
    n_int_anywhere = 0

    # initial state: N photons
    N_cur = N
    x = rng.uniform(-5,5,N)
    y = rng.uniform(-5,5,N)
    z = np.full(N, -100.0)
    # direction from source to (x,y,0)
    dx = x; dy = y; dz = 100.0
    norm = np.sqrt(dx*dx+dy*dy+dz*dz)
    ux = dx/norm; uy = dy/norm; uz = dz/norm
    E = np.where(rng.random(N)<0.5, 1.1732, 1.3325)
    alive = np.ones(N, bool)
    primary = np.ones(N, bool)
    pos = np.stack([x,y,z],axis=1)
    dire = np.stack([ux,uy,uz],axis=1)
    Ecut = 0.002

    it=0
    while alive.any():
        it+=1
        n_act = int(alive.sum())
        Em = np.maximum(E[alive], Ecut)
        mu = mu_tot(Em)
        s = -np.log(np.maximum(rng.random(n_act),1e-30))/mu
        p1 = pos[alive] + dire[alive]*s[:,None]
        # score cylinder: segment from pos to p1
        L = seg_in_cyl_vec(pos[alive], p1)
        # L nonzero -> score fluence
        m = L>0
        if m.any():
            Em_sub = Em[m]
            Lm = L[m]
            pri = primary[alive][m]
            ph_idx = np.clip(np.searchsorted(ph_edges, Em_sub, side='right')-1, 0, nph-1)
            ct_idx = np.clip(np.searchsorted(ct_edges, dire[alive][m][:,2], side='right')-1, 0, nct-1)
            np.add.at(flu_pri_ph, ph_idx[pri], Lm[pri])
            np.add.at(flu_sct_ph, ph_idx[~pri], Lm[~pri])
            np.add.at(flu_pri_ct, ct_idx[pri], Lm[pri])
            np.add.at(flu_sct_ct, ct_idx[~pri], Lm[~pri])
        # check outside cube
        outside = (p1[:,2] < SURF_Z) | (p1[:,2] > PH) | (np.abs(p1[:,0])>PH/2) | (np.abs(p1[:,1])>PH/2)
        # position update
        pos_new = np.where(alive[:,None], p1, pos)
        pos = pos_new
        # interaction point inside cube -> interact
        do_int = alive & ~outside
        n_int_anywhere += int(do_int.sum())
        # sample process
        comp_p = mu_comp(Em)/mu_tot(Em)
        coh_p  = mu_coh(Em)/mu_tot(Em)
        pho_p  = mu_pho(Em)/mu_tot(Em)
        # pair = rest
        r = rng.random(n_act)
        is_comp = r < comp_p
        r2 = (r-comp_p)/np.maximum(coh_p+pho_p+mu_pri(Em)/mu_tot(Em),1e-30)  # not exact; do sequential
        # sequential selection
        is_coh = (~is_comp) & (r < comp_p+coh_p)
        is_pho = (~is_comp) & (~is_coh) & (r < comp_p+coh_p+pho_p)
        is_pri = (~is_comp) & (~is_coh) & (~is_pho)
        
        # record interactions in cylinder before updating
        in_cyl = do_int & (L>0)
        if in_cyl.any():
            Em_in = Em[in_cyl]
            pri_in = primary[alive][in_cyl]
            comp_in = is_comp[alive][~outside][in_cyl]  # careful indexing: is_comp shape n_act corresp alive&? 
            # We built is_comp over alive photons (n_act). in_cyl index over alive photons too (L computed for alive). outside also over alive.
            # Let's index safely:
        # store per-alive arrays
        comp_a = is_comp; coh_a = is_coh; pho_a = is_pho; pri_a = is_pri
        incyl_a = do_int[alive] & (L>0)  # over alive-indexed arrays
        if incyl_a.any():
            Em_in = Em[incyl_a]
            pri_in = primary[alive][incyl_a]
            cp = comp_a[incyl_a]; ch = coh_a[incyl_a]; pp = pho_a[incyl_a]; pr = pri_a[incyl_a]
            # counts
            n_int['comp'] += int(cp.sum()); n_int['photo'] += int(pp.sum()); n_int['pair'] += int(pr.sum()); n_int['ray'] += int(ch.sum())
            pri_mask = pri_in
            n_int_pri['comp'] += int((cp&pri_mask).sum()); n_int_pri['photo'] += int((pp&pri_mask).sum()); n_int_pri['pair'] += int((pr&pri_mask).sum()); n_int_pri['ray'] += int((ch&pri_mask).sum())
            sct_mask = ~pri_in
            n_int_sct['comp'] += int((cp&sct_mask).sum()); n_int_sct['photo'] += int((pp&sct_mask).sum()); n_int_sct['pair'] += int((pr&sct_mask).sum()); n_int_sct['ray'] += int((ch&sct_mask).sum())
            n_scat_inter += int(sct_mask.sum())
            # energy transfer
            E_tr_comp = Em_in - Em_in/(1+(Em_in/mec2)*(1-0.0))  # placeholder; compute properly below with sampled cos
            # For transfer we need actual E' sampled; handle separately after sampling. For now approximate using mean? We'll do exact after angle sampling.
        # --- update photon states for interacting photons ---
        # we need to sample Compton cos and compute new direction for all comp
        act_idx = np.where(alive)[0]
        # For comp photons:
        if comp_a.any():
            Ecomp = Em[comp_a]
            rcomp = rng.random(int(comp_a.sum()))
            costh = sample_compton_cos(Ecomp, rcomp)
            Ep = Ecomp/(1+(Ecomp/mec2)*(1-costh))
            # new direction
            u_old = dire[alive][comp_a]
            # build basis
            # e1 = cross(u, z) normalized if |cross|>eps else x
            zvec = np.array([0,0,1.0])
            cross = np.cross(u_old, zvec)
            nc = np.linalg.norm(cross,axis=1)
            fallback = nc < 1e-8
            e1 = np.zeros_like(cross)
            e1[~fallback] = cross[~fallback]/nc[~fallback,None]
            e1[fallback] = np.array([1,0,0])
            e2 = np.cross(u_old, e1)
            phi = rng.uniform(0,TWOPI,int(comp_a.sum()))
            sinth = np.sqrt(np.maximum(1-costh*costh,0))
            u_new = u_old*costh[:,None] + (np.cos(phi)[:,None]*e1 + np.sin(phi)[:,None]*e2)*sinth[:,None]
            # electron recoil direction & energy
            p_e = Ecomp[:,None]*u_old - Ep[:,None]*u_new
            T_e = Ecomp - Ep
            pn = np.linalg.norm(p_e,axis=1)
            el_dir_rec = p_e/np.maximum(pn[:,None],1e-30)
            # energy transfer for compton = T_e
            # record in cylinder (do after)
            # update states
            set_idx = act_idx[comp_a]
            E[set_idx] = Ep
            dire[set_idx] = u_new
            primary[set_idx] = False
        # for coherent: update direction, keep E, primary=False
        if coh_a.any():
            Eco = Em[coh_a]
            r_coh = rng.random(int(coh_a.sum()))
            costh = sample_coherent_cos(Eco, r_coh)
            u_old = dire[alive][coh_a]
            cross = np.cross(u_old, np.array([0,0,1.0]))
            nc = np.linalg.norm(cross,axis=1)
            fallback = nc<1e-8
            e1 = np.zeros_like(cross)
            e1[~fallback] = cross[~fallback]/nc[~fallback,None]
            e1[fallback]=np.array([1,0,0])
            e2 = np.cross(u_old,e1)
            phi = rng.uniform(0,TWOPI,int(coh_a.sum()))
            sinth = np.sqrt(np.maximum(1-costh*costh,0))
            u_new = u_old*costh[:,None]+(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)*sinth[:,None]
            set_idx = act_idx[coh_a]
            dire[set_idx]=u_new
            primary[set_idx]=False
        # for photo: photon absorbed -> alive False
        # for pair: absorbed -> alive False
        die = do_int & (pho_a[alive[do_int]] if False else ((~is_comp)&(~is_coh)&(~is_pho))[alive])
        # simpler: set alive False where in_cube and (photo or pair)
        absorb = do_int & (~comp_a[alive]) & (~coh_a[alive])
        alive[absorb] = False
        # scattered photons that leave cube
        alive[outside] = False
        # low energy cutoff: treat as absorbed
        lowE = alive & (E < Ecut)
        alive[lowE] = False
        if it%200==0:
            print('iter',it,'alive',int(alive.sum()))
    return dict(flu_pri_ph=flu_pri_ph, flu_sct_ph=flu_sct_ph, flu_pri_ct=flu_pri_ct, flu_sct_ct=flu_sct_ct,
                n_int=n_int, n_int_pri=n_int_pri, n_int_sct=n_int_sct, n_scat_inter=n_scat_inter,
                E_tr_by_proc=E_tr_by_proc, E_tr_total=E_tr_total, E_tr_scat=E_tr_scat,
                el_ini_all=el_ini_all, el_ini_pri=el_ini_pri, el_dir=el_dir, n_el_all=n_el_all, n_el_pri=n_el_pri,
                N=N, t=time.time()-t0)

res = run_mc(20000, seed=1)
print('time', res['t'])
print('N',res['N'])
print('flu pri total (cm)', res['flu_pri_ph'].sum(), 'flu sct total', res['flu_sct_ph'].sum())
print('interactions', res['n_int'], 'scat inter', res['n_scat_inter'])
