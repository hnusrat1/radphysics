code = """
import numpy as np
from scipy.interpolate import interp1d
import json, sys

np.random.seed(42)

m_e = 0.51099895
r_e = 2.8179403262e-13
NA = 6.02214076e23
SSD = 100.0; fh = 5.0; zs = 0.0; zc = 5.0; zh = 0.1
zlo = zc - zh; zhi = zc + zh; Rc = 1.0
th = np.arctan(fh/SSD); Opyr = 4.0*np.arcsin(np.sin(th)**2)
ne = 10.0*NA/18.015

print(f"O={Opyr:.6f} ne={ne:.4e}", file=sys.stderr)

ea = np.array([1.0,1.1,1.1732,1.2,1.25,1.3,1.3325,1.4,1.5])
xT = np.array([0.07070,0.06780,0.06590,0.06520,0.06400,0.06280,0.06220,0.06080,0.05780])
xC = np.array([0.06950,0.06650,0.06430,0.06350,0.06200,0.06060,0.06000,0.05850,0.05550])
xP = np.array([0.00040,0.00030,0.00025,0.00022,0.00020,0.00017,0.00015,0.00013,0.00010])
xPP= np.array([0.0,0.0,0.00020,0.00028,0.00050,0.00080,0.00100,0.00130,0.00150])
xR = np.array([0.00080,0.00068,0.00060,0.00055,0.00050,0.00043,0.00040,0.00035,0.00030])

fT=interp1d(ea,xT,kind='cubic',bounds_error=False,fill_value='extrapolate')
fC=interp1d(ea,xC,kind='cubic',bounds_error=False,fill_value='extrapolate')
fP=interp1d(ea,xP,kind='cubic',bounds_error=False,fill_value='extrapolate')
fPP=interp1d(ea,xPP,kind='cubic',bounds_error=False,fill_value='extrapolate')
fR=interp1d(ea,xR,kind='cubic',bounds_error=False,fill_value='extrapolate')

def mu_t(E): return float(fT(np.clip(E,0.01,10.0)))
def mu_c(E): return float(fC(np.clip(E,0.01,10.0)))
def mu_ph(E): return float(fP(np.clip(E,0.01,10.0)))
def mu_pp(E): return float(fPP(np.clip(E,0.01,10.0)))
def mu_rl(E): return float(fR(np.clip(E,0.01,10.0)))

def c_E(E,ct): return E/(1.0+(E/m_e)*(1.0-ct))
def KN(E,ct):
    thv=np.arccos(np.clip(ct,-1.,1.)); Eo=c_E(E,ct)
    return (r_e**2/2.0)*(Eo/E)**2*(E/Eo+Eo/E-np.sin(thv)**2)

def sample_KN(E):
    mx=r_e**2*1.05
    while True:
        ct=np.random.uniform(-1.,1.)
        if np.random.random()*mx<KN(E,ct): break
    Eo=c_E(E,ct); te=E-Eo
    num=E-Eo*ct; den=np.sqrt(E**2+Eo**2-2*E*Eo*ct)
    ce=num/den if den>0 else 1.0
    return Eo,ct,te,ce

def rot(v,ct):
    u=v/np.linalg.norm(v); thv=np.arccos(np.clip(ct,-1.,1.))
    ph=np.random.uniform(0.,2.*np.pi)
    if abs(u[2])<0.999: w=np.cross(u,np.array([0.,0.,1.]))
    else: w=np.cross(u,np.array([1.,0.,0.]))
    w=w/np.linalg.norm(w); vv=np.cross(w,u)
    return u*ct+w*np.sin(thv)*np.cos(ph)+vv*np.sin(thv)*np.sin(ph)

def src_samp():
    E=1.1732 if np.random.random()<0.5 else 1.3325
    x=np.random.uniform(-fh,fh); y=np.random.uniform(-fh,fh)
    d=np.array([x,y,SSD]); d=d/np.linalg.norm(d)
    return E,d,x,y

def sample_proc(E):
    t=mu_c(E)+mu_ph(E)+mu_pp(E)+mu_rl(E); r=np.random.random()*t
    if r<mu_c(E): return 'compton'
    elif r<mu_c(E)+mu_ph(E): return 'photoelectric'
    elif r<mu_c(E)+mu_ph(E)+mu_pp(E): return 'pair'
    else: return 'rayleigh'

# ===== PRIMARY RATE =====
NMC = 500000
pt = 0.0
pb = {'compton':0.,'photoelectric':0.,'pair':0.,'rayleigh':0.}
for _ in range(NMC):
    E,d,x,y = src_samp()
    dz = d[2]
    if dz<=0: continue
    x_ep = x*(SSD+zlo)/SSD
    y_ep = y*(SSD+zlo)/SSD
    if np.sqrt(x_ep**2+y_ep**2)>Rc: continue
    path = 0.2/dz
    muE = mu_t(E)
    t_sc = (zlo-zs)/dz
    att = np.exp(-muE*t_sc)
    p_int = 1.0-np.exp(-muE*path)
    w = att*p_int
    pt += w
    mt = mu_c(E)+mu_ph(E)+mu_pp(E)+mu_rl(E)
    pb['compton'] += w*mu_c(E)/mt
    pb['photoelectric'] += w*mu_ph(E)/mt
    pb['pair'] += w*mu_pp(E)/mt
    pb['rayleigh'] += w*mu_rl(E)/mt
pt /= NMC
for k in pb: pb[k] /= NMC
print(f"PRIMARY: {pt:.8f}", file=sys.stderr)
for k,v in pb.items(): print(f"  {k}: {v:.10f}", file=sys.stderr)

# ===== SCRATE =====
st = 0.0
sb = {'compton':0.,'photoelectric':0.,'pair':0.,'rayleigh':0.}
for _ in range(NMC):
    E,d,x,y = src_samp()
    dz = d[2]; muE = mu_t(E)
    if dz<=0: continue
    t_max = 30.0/dz
    C = 1.0-np.exp(-muE*t_max)
    if C<=0: continue
    # scatter point on ray from source
    t_scat = -np.log(1.0-np.random.random()*C)/muE
    sp = np.array([x,y,zs]) + t_scat*d
    if abs(sp[0])>15 or abs(sp[1])>15: continue
    pdf_scat = muE*np.exp(-muE*t_scat)/C
    # forced direction to scoring volume center
    d2v = np.array([0.,0.,zc]) - sp
    dist = np.linalg.norm(d2v)
    if dist<0.01: continue
    d2v = d2v/dist
    costh = np.dot(d,d2v)
    costh = np.clip(costh,-1.,1.)
    dcs = KN(E,costh)
    E_out = c_E(E,costh)
    # solid angle of scoring volume
    cosv = abs(d2v[2])
    proj = np.pi*Rc**2*max(cosv,0.01)
    solid = proj/(dist**2)
    # check ray hits scoring volume
    it = intersect_cyl(sp,d2v,Rc,zlo,zhi)
    if it is None: continue
    te, tx = it
    if te<0: te=0
    pv = tx-te
    if pv<=0: continue
    mu_out = mu_t(E_out)
    att_v = np.exp(-mu_out*te)
    p_int_v = 1.0-np.exp(-mu_out*pv)
    w = ne*dcs*solid*att_v*p_int_v/pdf_scat
    st += w
    mt = mu_c(E_out)+mu_ph(E_out)+mu_pp(E_out)+mu_rl(E_out)
    sb['compton'] += w*mu_c(E_out)/mt
    sb['photoelectric'] += w*mu_ph(E_out)/mt
    sb['pair'] += w*mu_pp(E_out)/mt
    sb['rayleigh'] += w*mu_rl(E_out)/mt
st /= NMC
for k in sb: sb[k] /= NMC
print(f"SCATTER: {st:.8f}", file=sys.stderr)
for k,v in sb.items(): print(f"  {k}: {v:.10f}", file=sys.stderr)

total = pt+st
frac_s = st/total
print(f"Total: {total:.8f}, scatter fraction: {frac_s:.4f}", file=sys.stderr)

# ===== GENERATE 200 EVENTS =====
n_prim = int(round(200*pt/total))
n_scat = 200 - n_prim
print(f"Generating {n_prim} primary + {n_scat} scatter events", file=sys.stderr)

events = []

# Primary events
gen = 0; atts = 0; max_a = 5000000
while gen < n_prim and atts < max_a:
    atts += 1
    E,d,x,y = src_samp()
    dz = d[2]
    if dz<=0: continue
    x_ep = x*(SSD+zlo)/SSD
    y_ep = y*(SSD+zlo)/SSD
    if np.sqrt(x_ep**2+y_ep**2)>Rc: continue
    path = 0.2/dz
    muE = mu_t(E)
    t_sc = (zlo-zs)/dz
    att = np.exp(-muE*t_sc)
    p_int = 1.0-np.exp(-muE*path)
    # accept/reject
    max_p = np.exp(-0.064*5.0)*(1.0-np.exp(-0.064*0.2))
    if np.random.random()*max_p*1.5 > att*p_int: continue
    
    proc = sample_proc(E)
    if proc == 'compton':
        Eo,ct,te,ce = sample_KN(E)
        events.append({'process':'compton','e_in_MeV':E,'e_out_MeV':Eo,'cos_scatter':ct,'t_e_MeV':te,'cos_electron':ce,'t_positron_MeV':None})
        gen += 1
    elif proc == 'photoelectric':
        ce = np.random.uniform(0.7,1.0)
        events.append({'process':'photoelectric','e_in_MeV':E,'e_out_MeV':None,'cos_Scatter':None,'t_e_MeV':E,'cos_electron':ce,'t_positron_MeV':None})
        gen += 1
    elif proc == 'pair':
        tk = E - 2.0*m_e
        if tk > 0:
            te = np.random.uniform(0.,tk)
            tp = tk - te
            ce = np.random.uniform(0.9,1.0)
            events.append({'process':'pair','e_inMeV':E,'e_out_MeV':None,'cos_scatter':None,'t_e_MeV':te,'cos_electron':ce,'t_positron_MeV':tp})
            gen += 1

print("Ended", file=sys.stderr)
"""
exec(code)