
ana = r'''
import sys, pickle, numpy as np
sys.path.insert(0,'/tmp/work')
import phys, mc
from phys import *
R = pickle.load(open('/tmp/work/res_t1.pkl','rb'))
N = R['N']; V = mc.V_POI; OM = mc.OMEGA; PREF = OM*mc.DIST**2/N   # per-weight track factor
CONV = 1.602176634e-10   # MeV/g -> Gy
LINES = mc.LINES
rng = np.random.default_rng(7)

# ---------- primary (analytic) ----------
mu_L = mu_tot(LINES)
zavg = lambda mu: np.exp(-5*mu)*np.sinh(0.1*mu)/(0.1*mu)
phiP = 0.5*zavg(mu_L)            # per line, per Phi0 (average over POI volume)
PRIM_FLU = phiP.sum()
TVp = phiP*V                     # track length in V per Phi0 for each line
cnt_p = {}; etr_p={}
cnt_p['inc'] = np.sum(TVp*mu_inc(LINES)); cnt_p['pe']=np.sum(TVp*mu_pe(LINES)); cnt_p['pair']=np.sum(TVp*mu_pair(LINES)); cnt_p['coh']=np.sum(TVp*mu_coh(LINES))
etr_p['inc'] = np.sum(TVp*mu_inc(LINES)*LINES*(1-eps_mean(LINES)))
etr_p['pe'] = np.sum(TVp*mu_pe(LINES)*LINES)
etr_p['pair'] = np.sum(TVp*mu_pair(LINES)*np.maximum(LINES-1.022,0))
etr_p['coh']=0.0

# primary births (analytic sampling via KN)
Np = 400000
pb_T=[]; pb_w=[]; pb_proc=[]; pb_cos=[]; pb_sign=[]
for L,tv in zip(LINES, TVp):
    E = np.full(Np, L)
    eps, cth = kn_sample(E, rng)
    T = L*(1-eps)
    phi = 2*np.pi*rng.random(Np)
    d = np.array([0,0,1.0])
    dsc = rotate_dir(np.tile(d,(Np,1)), cth, phi)
    pv = np.tile(d,(Np,1)) - eps[:,None]*dsc
    pv /= np.linalg.norm(pv,axis=1)[:,None]
    pb_T.append(T); pb_w.append(np.full(Np, tv*mu_inc(L)/Np)); pb_proc.append(np.zeros(Np,int)); pb_cos.append(pv[:,2]); pb_sign.append(np.zeros(Np,int))
    # pe
    pb_T.append(np.array([L])); pb_w.append(np.array([tv*mu_pe(L)])); pb_proc.append(np.array([1])); pb_cos.append(np.array([1.0])); pb_sign.append(np.array([0]))
    # pair: e- and e+ uniform sharing
    Tp = max(L-1.022,0)
    fr = rng.random(Np)
    for sgn in [0,1]:
        pb_T.append(Tp*(fr if sgn==0 else 1-fr)); pb_w.append(np.full(Np, tv*mu_pair(L)/Np)); pb_proc.append(np.full(Np,2)); pb_cos.append(np.ones(Np)); pb_sign.append(np.full(Np,sgn))
pb_T=np.concatenate(pb_T); pb_w=np.concatenate(pb_w); pb_proc=np.concatenate(pb_proc); pb_cos=np.concatenate(pb_cos); pb_sign=np.concatenate(pb_sign)

# scattered births (MC, already per-Phi0 weights)
sb_T=R['birth_T']; sb_w=R['birth_w']; sb_proc=R['birth_proc']; sb_cos=R['birth_cos']; sb_sign=R['birth_sign']
# note: MC weights carry PREF already (track factor) 
# counts
print('N births scattered', sb_T.size, 'sum w', sb_w.sum(), 'primary births sum w', pb_w.sum())
print('scattered inc count', R['cnt']['inc'], 'primary inc', cnt_p['inc'])

# ---------- L1 ----------
Tsc = R['Tsc']
PE_EDGES = mc.PE_EDGES
Phi_s_bin = PREF*Tsc/V          # scattered fluence per bin per Phi0
Phi_p_bin = np.zeros(28)
for L,p in zip(LINES,phiP):
    k = np.searchsorted(PE_EDGES, L, side='right')-1
    Phi_p_bin[k] += p
PHI_S = Phi_s_bin.sum()
Esum_s = R['Esum_w']*PREF/PREF/V  # Esum_w already includes PREF
Esum_s = R['Esum_w']/V
Esum_p = np.sum(phiP*LINES)
L1_total = PRIM_FLU + PHI_S
L1_scat_frac = PHI_S/L1_total
L1_scat_E_frac = Esum_s/(Esum_s+Esum_p)
L1_mean_E = (Esum_s+Esum_p)/L1_total
Co = R['Tco']; ang_s = Co/Co.sum()
ang_p = np.zeros(10); ang_p[9]=1.0

# ---------- L2 ----------
cnt_s = R['cnt']; etr_s = R['etr']
def fracs(c): 
    tot = sum(c.values()); return {k: c[k]/tot for k in ['inc','pe','pair','coh']}
c_all = {k: cnt_s[k]/V*0 + cnt_s[k] + cnt_p[k] for k in cnt_s}  # both in per-Phi0 track units (T_V)
# scattered cnt in R are sum(pref*w*l*mu) -> track-length-weighted counts (not per volume)
cnt_s_T = cnt_s
L2_int_all = {k: cnt_s[k]+cnt_p[k] for k in cnt_s}
L2_int_prim = {k: cnt_p[k] for k in cnt_s}
L2_int_scat = {k: cnt_s[k] for k in cnt_s}
L2_etr_all = {k: etr_s[k]+etr_p[k] for k in cnt_s}
def tofrac(d):
    tot=sum(d.values()); return {'compton':d['inc']/tot,'photoelectric':d['pe']/tot,'pair':d['pair']/tot,'rayleigh':d['coh']/tot}
L2_frac = tofrac(L2_int_all)
L2_etr = tofrac(L2_etr_all)
L2_prim = tofrac(L2_int_prim)
L2_scat = tofrac(L2_int_scat)
L2_scat_share = sum(L2_int_scat.values())/sum(L2_int_all.values())

# ---------- L3 ----------
EE = mc.EE_EDGES; CO = mc.CO_EDGES
allT = np.concatenate([sb_T,pb_T]); allw = np.concatenate([sb_w,pb_w])
allcos = np.concatenate([sb_cos,pb_cos]); allsign = np.concatenate([sb_sign,pb_sign])
allproc = np.concatenate([sb_proc,pb_proc])
def binned(T,w,edges):
    h,_ = np.histogram(T,bins=edges,weights=w); return h
h3 = binned(allT,allw,EE); L3_spec = h3/h3.sum()
hp = binned(pb_T,pb_w,EE); L3_spec_p = hp/hp.sum()
L3_mean = np.sum(allT*allw)/np.sum(allw)
L3_mean_p = np.sum(pb_T*pb_w)/np.sum(pb_w)
hd,_ = np.histogram(allcos,bins=CO,weights=allw); L3_dir = hd/hd.sum()

# ---------- electron model ----------
ESTAR = np.loadtxt('/tmp/data/estar_water.txt')
Te, Scol_w, Srad_w, Stot_w, Rcsda_w, Yrad, dens = ESTAR.T
def interpE(x, y, T):
    return np.exp(np.interp(np.log(T), np.log(x), np.log(np.maximum(y,1e-30))))
MEC2 = M_E
# fine grid
nT = 240
Tedges = np.logspace(np.log10(0.01), np.log10(1.2), nT+1)
Tnode = np.sqrt(Tedges[:-1]*Tedges[1:]); dT = np.diff(Tedges)
# Q0 from e- births (per volume, per MeV, per Phi0)
sel = allsign==0
q0_hist,_ = np.histogram(allT[sel], bins=Tedges, weights=allw[sel])
Q0 = q0_hist/V/dT
# primary-only Q0 (for separate delta fraction, not needed) 
Scol_n = interpE(Te,Scol_w,Tnode)
def dsig_deltaE(Ep, eps):
    """Moller dsigma/deps (cm^2/MeV) for primary kinetic energy Ep (MeV), delta energy eps (MeV)"""
    tau = Ep/MEC2
    beta2 = tau*(tau+2)/(tau+1)**2
    x = eps/Ep
    F = 1/x**2 + 1/(1-x)**2 + (tau/(tau+1))**2 - (2*tau+1)/(tau+1)**2/(x*(1-x))
    K = 2*np.pi*RE**2*MEC2/beta2
    return K*F/Ep**2
NEL = NA*ZA_W   # electrons per gram (water, rho=1)
def sigma_delta_above(Ep, D=0.01):
    """number-per-length: n*int_D^{Ep/2} dsig/deps d eps (per cm)"""
    e = np.linspace(D, Ep/2, 2001)
    return NEL*np.trapezoid(dsig_deltaE(Ep,e), e)

# solve: Phi = (1/S) int_E^inf (Q0 + Qd), Qd from Phi
Phi = np.zeros(nT); Phid = np.zeros(nT)
Qd = np.zeros(nT)
# precompute delta kernel matrix: K[k,j] = NEL * dsig(E_j -> T_k) * dT_j for E_j>=2T_k
Kmat = np.zeros((nT,nT))
for j in range(nT):
    Ej = Tnode[j]
    mask = Tnode <= Ej/2
    Kmat[mask, j] = NEL*dsig_deltaE(Ej, Tnode[mask])*dT[j]
def solve_phi(Q):
    cum = np.zeros(nT+1)
    # integral from node k to top: sum_{j>k} Q_j dT_j + 0.5 Q_k dT_k
    tail = np.concatenate([np.cumsum((Q*dT)[::-1])[::-1][1:], [0.0]])
    I = tail + 0.5*Q*dT
    return I/ Scol_n
Q = Q0.copy()
for it in range(60):
    Phi_new = solve_phi(Q)
    Qd_new = Kmat @ Phi_new
    Q = Q0 + Qd_new
    if it>0 and np.max(np.abs(Phi_new-Phi))<1e-10*np.max(np.abs(Phi_new)): 
        Phi = Phi_new; break
    Phi = Phi_new
Phi_tot = solve_phi(Q)
Phi_0 = solve_phi(Q0)
Phi_d = Phi_tot - Phi_0
# coarse bins
Tc = EE
L4_bin = np.zeros(25); L4_bin_d = np.zeros(25)
for k in range(25):
    m = (Tnode>=Tc[k])&(Tnode<Tc[k+1])
    L4_bin[k] = np.sum(Phi_tot[m]*dT[m]); L4_bin_d[k]=np.sum(Phi_d[m]*dT[m])
L4_fluence = L4_bin.sum()
L4_delta_frac = L4_bin_d.sum()/L4_fluence
L4_mean_E = np.sum(Phi_tot*dT*Tnode)/np.sum(Phi_tot*dT)
Srad_n = interpE(Te,Srad_w,Tnode)
L4_rad_coll = np.sum(Phi_tot*dT*Srad_n)/np.sum(Phi_tot*dT*Scol_n)
# check: electron fluence above 10 keV from births (track-length consistency)
R_n = interpE(Te,Rcsda_w,Tnode)
def Rc(T): return interpE(Te,Rcsda_w,T)
print('L4 fluence (per Phi0)', L4_fluence, 'delta frac', L4_delta_frac, 'mean E', L4_mean_E, 'rad/coll', L4_rad_coll)
print('check births-range fluence', np.sum(allw[sel]*(Rc(allT[sel])-Rc(0.01)))/V)

# ---------- L4 angular (approximate MC of track-length weights) ----------
# primary-origin electrons: weight w*(R(T)-R(0.01)), cos from birth
wr = allw[sel]*(Rc(allT[sel])-Rc(0.01))
c_pop = allcos[sel]
# delta population: parent energies from Phi_tot*dT with delta production
prod = Phi_tot*dT*np.array([sigma_delta_above(E) for E in Tnode])*NEL/NEL
# sample deltas
nd = 200000
pj = prod/prod.sum()
jj = rng.choice(nT, size=nd, p=pj)
Wd = prod.sum()/nd  # per-delta weight (total delta production rate per Phi0 normalised to V?)
# delta energies from dsigma/deps on [0.01, E/2]
eps_s = np.empty(nd); cos_d = np.empty(nd)
for k in range(nT):
    m = np.nonzero(jj==k)[0]
    if m.size==0: continue
    Ep = Tnode[k]
    e = np.geomspace(0.01, Ep/2, 800)
    f = dsig_deltaE(Ep, e)*e
    cdf = np.concatenate([[0],np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(e))]); cdf/=cdf[-1]
    u = rng.random(m.size)
    eps_s[m] = np.interp(u, cdf, e)
    tau=Ep/MEC2
    cos_d[m] = np.sqrt(np.clip(eps_s[m]*(tau+2)/(tau*(eps_s[m]/MEC2+2)*1.0),0,1)) if False else np.sqrt(np.clip(eps_s[m]*(Ep+2*MEC2)/(Ep*(eps_s[m]+2*MEC2)),0,1))
# parent direction cos sampled from primary-origin births
par_cos = c_pop[rng.integers(0,c_pop.size,nd)] if False else rng.choice(c_pop, size=nd)
cos_child = par_cos*cos_d
wd = Wd*(Rc(eps_s)-Rc(0.01))
Hall,_ = np.histogram(np.concatenate([c_pop,cos_child]), bins=CO, weights=np.concatenate([wr, wd]))
L4_ang = Hall/Hall.sum()
print('delta-based births-per-Phi0-ratio: Wd*nd', Wd*nd, 'sum prod', prod.sum())

# ---------- L5 ----------
# kerma/collision kerma per Phi0
K_s = (R['etr']['inc']+R['etr']['pe']+R['etr']['pair'])/V*CONV
K_p = np.sum(phiP*LINES*(mu_inc(LINES)*(1-eps_mean(LINES))+mu_pe(LINES)+mu_pair(LINES)*np.maximum(1-1.022/LINES,0)))*CONV
K_tot = K_s + K_p
# births-based: sum w*T and w*T*(1-Y)
Yn = interpE(Te,Yrad,allT) if False else np.interp(np.log(allT), np.log(Te), Yrad)
sumT = np.sum(allw*allT)
sumTY = np.sum(allw*allT*Yn)
g = sumTY/sumT
Kcol_s = np.sum(sb_w*sb_T*(1-np.interp(np.log(sb_T), np.log(Te), Yrad)))/V*CONV
Kcol_p = np.sum(pb_w*pb_T*(1-np.interp(np.log(pb_T), np.log(Te), Yrad)))/V*CONV
Kcol_all = Kcol_s+Kcol_p
K_births = sumT/V*CONV
print('K from etr',K_tot,'K births',K_births,'Kcol', Kcol_all, 'g', g, 'K_s',K_s,'K_p',K_p)
parent_scat_frac = Kcol_s/Kcol_all
D_over = Kcol_all   # CPE: D = K_col
beta = 1.0

# ---------- Spencer-Attix water/air ----------
I_w=75.0e-6; I_a=85.7e-6; ZAa=0.49919; ZAw=0.55508
def bethe(T, I, ZA):
    tau = T/MEC2
    b2 = tau*(tau+2)/(tau+1)**2
    Fm = 1-b2 + (tau**2/8 - (2*tau+1)*np.log(2))/(tau+1)**2
    return 2*np.pi*RE**2*MEC2*NA*ZA/b2*(np.log(tau**2*(tau+2)/(2*(I/MEC2)**2)) + Fm)
# check bethe water vs ESTAR
chk = bethe(Tnode, I_w, ZAw)
print('bethe/ESTAR water (MeV cm2/g) at 0.1,1 MeV:', bethe(np.array([0.1,1.0]),I_w,ZAw), np.interp(np.log([0.1,1.0]),np.log(Te),Scol_w))
S_air = Scol_n * bethe(Tnode,I_a,ZAa)/bethe(Tnode,I_w,ZAw)
# restricted with Delta=0.01 MeV
Dl = 0.01
def delta_loss(Ep, n_per_g):
    e = np.linspace(Dl, Ep/2, 2001)
    return n_per_g*np.trapezoid(e*dsig_deltaE(Ep,e), e)
dl_w = np.array([delta_loss(E, NA*ZAw) for E in Tnode])
dl_a = np.array([delta_loss(E, NA*ZAa) for E in Tnode])
Sres_w = Scol_n - dl_w
Sres_a = S_air - dl_a
Sres_w_D = interpE(Te,Scol_w,np.array([Dl]))[0] - delta_loss(Dl, 0)  # not used
# value at Delta: restricted at Dl
Sres_w_D = np.interp(np.log(Dl), np.log(Te), Scol_w)  # unrestricted at Delta (approx)
Sres_a_D = np.interp(np.log(Dl), np.log(Te), Scol_w)*bethe(np.array([Dl]),I_a,ZAa)[0]/bethe(np.array([Dl]),I_w,ZAw)[0]
m = Tnode>=Dl
PhiS = Phi_tot
# track-end term: Delta*Phi(Delta)*S_res(Delta): Phi at Delta = Phi_tot at lowest node
Phi_D = np.interp(np.log(Dl), np.log(Tnode), Phi_tot)
num = np.sum(PhiS[m]*Sres_w[m]*dT[m]) + Dl*Phi_D*Sres_w_D
den = np.sum(PhiS[m]*Sres_a[m]*dT[m]) + Dl*Phi_D*Sres_a_D
sw_air = num/den
print('s_w_air', sw_air, 'num',num,'den',den)

# ---------- depth dose (scattered slabs + primary analytic) ----------
Kdep = R['Kdep']*PREF/(np.pi*1.0**2*0.25)*CONV
zc = mc.DZ_CENT
mu_L_all = mu_L
prim_K = np.zeros(zc.size)
for L,mu,ep in zip(LINES, mu_L, phiP*0+0.5):
    fl = 0.5*(105.0/(100.0+zc))**2*np.exp(-mu*zc)
    prim_K += fl*L*(mu_inc(L)*(1-eps_mean(L)) + mu_pe(L) + mu_pair(L)*max(1-1.022/L,0))*CONV
Ktot_z = Kdep + prim_K
i5 = np.argmin(np.abs(zc-5.0)); imax = np.argmax(Ktot_z)
PDD5 = 100*Ktot_z[i5]/Ktot_z[imax]
print('zmax', zc[imax], 'PDD5', PDD5, 'K(z) scat/prim at 0.5,5:', Kdep[1], prim_K[1], Kdep[19], prim_K[19])

out = dict(
 L1_primary_fluence_per_phi0 = PRIM_FLU,
 L1_total_fluence_per_phi0 = L1_total,
 L1_scattered_fraction_fluence = L1_scat_frac,
 L1_scattered_fraction_energy_fluence = L1_scat_E_frac,
 L1_mean_photon_energy_MeV = L1_mean_E,
 L1_photon_fluence_spectrum_primary = Phi_p_bin.tolist(),
 L1_photon_fluence_spectrum_scattered = Phi_s_bin.tolist(),
 L1_angular_primary = ang_p.tolist(),
 L1_angular_scattered = ang_s.tolist(),
 L2_interaction_fractions = L2_frac,
 L2_energy_transfer_fractions = L2_etr,
 L2_interaction_fractions_primary = L2_prim,
 L2_interaction_fractions_scattered = L2_scat,
 L2_scattered_share_of_interactions = L2_scat_share,
 L3_initial_spectrum = L3_spec.tolist(),
 L3_initial_spectrum_parent_primary = L3_spec_p.tolist(),
 L3_mean_initial_energy_MeV = L3_mean,
 L3_mean_initial_energy_parent_primary_MeV = L3_mean_p,
 L3_initial_direction = L3_dir.tolist(),
 L4_electron_fluence_per_phi0 = L4_fluence,
 L4_electron_fluence_spectrum = L4_bin.tolist(),
 L4_delta_fraction = L4_delta_frac,
 L4_mean_electron_energy_MeV = L4_mean_E,
 L4_angular = L4_ang.tolist(),
 L4_radiative_to_collision = L4_rad_coll,
 L5_D_over_phi0_Gy_cm2 = D_over,
 L5_K_over_phi0_Gy_cm2 = K_tot,
 L5_Kcol_over_phi0_Gy_cm2 = Kcol_all,
 L5_beta = beta,
 L5_g = 1-Kcol_all/K_tot,
 L5_dose_fraction_parent_scattered = parent_scat_frac,
 L5_s_w_air = sw_air,
 L5_PDD5_percent = PDD5,
)
pickle.dump(out, open('/tmp/work/answers.pkl','wb'))
import json
def conv(o):
    if isinstance(o,dict): return {k:conv(v) for k,v in o.items()}
    if isinstance(o,list): return [conv(x) for x in o]
    if isinstance(o,(np.floating,float)): return float(o)
    if isinstance(o,(np.integer,int)): return int(o)
    return o
print(json.dumps(conv(out), indent=0)[:6000])
'''
open('/tmp/work/ana.py','w').write(ana)
import subprocess
r=subprocess.run("cd /tmp/work && python ana.py",shell=True,capture_output=True,text=True)
print(r.stdout[-8000:], r.stderr[-3000:])
