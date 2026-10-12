
import numpy as np, math

# reload MC results
d=np.load('mc_results.npz')
tl_p=d['tl_p']; tl_s=d['tl_s']; tlp_c=d['tlp_c']; tls_c=d['tls_c']
sumE_p=float(d['sumE_p']); sumE_s=float(d['sumE_s'])
sumLEmu_p=float(d['sumLEmu_p']); sumLEmu_s=float(d['sumLEmu_s'])
Kdep=d['Kdep']; Kdep_s=d['Kdep_s']; depths=d['depths']
M=2200000.0
Om=4*math.atan(50.0/(100.0*math.sqrt(40000+100+100)))  # = same as before
V=math.pi*1.0*0.2
C=(Om/M)*(105.0**2)/V
print("C",C)

Etab = np.array([0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0],float)
mu_tab   = np.array([0.2269,0.2059,0.1838,0.1707,0.1505,0.1373,0.1186,0.1061,0.0969,0.0896,0.0787,0.0707,0.0632,0.0576,0.0494],float)
mu_coh   = np.array([0.0222,0.0193,0.0161,0.0154,0.0118,0.0097,0.0073,0.0061,0.0053,0.0047,0.0039,0.0033,0.0028,0.0024,0.0019],float)
mu_en_tab= np.array([0.0406,0.0323,0.0260,0.0256,0.0277,0.0297,0.0319,0.0328,0.0330,0.0329,0.0320,0.0309,0.0297,0.0285,0.0262],float)
def interp1(x, xe, ye):
    return float(np.exp(np.interp(np.log(x), np.log(xe), np.log(ye))))
def mu_total(E): return interp1(E,Etab,mu_tab)
def mu_re(E): return interp1(E,Etab,mu_coh)
def mu_en(E): return interp1(E,Etab,mu_en_tab)
def mu_pe(E): return 0.0064*(E/0.1)**(-3.2)
def mu_pair(E):
    if E<=1.022: return 0.0
    return 0.00025 + 0.00015*(E-1.25)
def comp_frac(E):
    m=mu_total(E); c=mu_re(E); p=mu_pe(E); q=mu_pair(E)
    comp=m-c-p-q
    if comp<0: comp=m*0.85
    return comp,m,c,p,q

rng=np.random.default_rng(7)
def kn_sample(alpha, N):
    out=[]
    while len(out)<N:
        mu=rng.uniform(-1,1)
        r=1.0/(1.0+alpha*(1.0-mu))
        f=r*r*(r+1.0/r-(1.0-mu*mu))
        if rng.random()*2.0<f:
            out.append(mu)
    return np.array(out)

edges_ph=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
edges_e=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
edges_c=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

prim_phi = C*tl_p
scat_phi = C*tl_s

# primary lines
E1=1.1732; E2=1.3325
F1=prim_phi[23]; F2=prim_phi[26]
print("primary line fluences per Phi0:",F1,F2,"sum",F1+F2)

# scattered bin midpoints
Emid=(edges_ph[:-1]+edges_ph[1:])/2

# ---------------- L2 analytic ----------------
def process_weights(E, Phi):
    comp,m,c,p,q=comp_frac(E)
    return Phi*comp, Phi*c, Phi*p, Phi*q, Phi*m

def avg_compton_Te(E, nsamp=2000):
    alpha=E/0.511
    mu=kn_sample(alpha,nsamp)
    Ep=E/(1.0+alpha*(1.0-mu))
    return float(np.mean(E-Ep))

# primary
ip={k:0.0 for k in ['comp','pe','pair','ray']}; it=ip.copy(); im=0.0
for E,F in [(E1,F1),(E2,F2)]:
    comp,m,c,p,q=comp_frac(E)
    im+=F*m
    ip['comp']+=F*comp; ip['pe']+=F*p; ip['pair']+=F*q; ip['ray']+=F*c
    TeC=avg_compton_Te(E)
    it['comp']+=F*comp*TeC; it['pe']+=F*p*E; it['pair']+=F*q*(E-1.022); it['ray']+=0.0
L2p_int={k:ip[k]/im for k in ip}
L2p_tr ={k: (it[k]/sum(it.values())) for k in it} if sum(it.values())>0 else it
print("L2 primary interaction frac",L2p_int)
print("L2 primary transfer frac",L2p_tr)

# scattered
is_={k:0.0 for k in ['comp','pe','pair','ray']}; ist=is_.copy(); ims=0.0
for E,F in zip(Emid, scat_phi):
    if F<=0: continue
    comp,m,c,p,q=comp_frac(E)
    ims+=F*m
    is_['comp']+=F*comp; is_['pe']+=F*p; is_['pair']+=F*q; is_['ray']+=F*c
    if comp>0:
        TeC=avg_compton_Te(E)
        ist['comp']+=F*comp*TeC
    ist['pe']+=F*p*E
    if E>1.022: ist['pair']+=F*q*(E-1.022)
    ist['ray']+=0.0
L2s_int={k:is_[k]/ims for k in is_}
L2s_tr={k:ist[k]/sum(ist.values()) for k in ist} if sum(ist.values())>0 else ist
print("L2 scattered interaction frac",L2s_int)
print("L2 scattered transfer frac",L2s_tr)

# all
ia={k:ip[k]+is_[k] for k in ip}; iat={k:it[k]+ist[k] for k in it}
imat=im+ims
L2_int={k:ia[k]/imat for k in ia}
L2_tr={k:iat[k]/sum(iat.values()) for k in iat}
print("L2 all interaction frac",L2_int)
print("L2 all transfer frac",L2_tr)
print("L2_scattered_share_of_interactions",ims/imat)

# ---------------- L3 analytic ----------------
nb_e=25; nb_c=10
h_all=np.zeros(nb_e); h_prim=np.zeros(nb_e)
dir_all=np.zeros(nb_c); dir_prim=np.zeros(nb_c)
wtot=0.0; wprim=0.0
wE=0.0; wEp=0.0
N=300
def ebindex(val):
    i=int(np.searchsorted(edges_e,val,side='right')-1)
    i=max(0,min(nb_e-1,i)); return i
def cbindex(val):
    i=int(np.searchsorted(edges_c,val,side='right')-1)
    i=max(0,min(nb_c-1,i)); return i

def add_source(E, Phi, primflag):
    global wtot,wprim,wE,wEp
    if Phi<=0: return
    comp,m,c,p,q=comp_frac(E)
    # Compton electrons
    if comp>0 and Phi>0:
        alpha=E/0.511
        mu=kn_sample(alpha,N)
        Ep=E/(1.0+alpha*(1.0-mu))
        Te=E-Ep
        pe=np.sqrt(E*E+Ep*Ep-2*E*Ep*mu)
        cose=(E-Ep*mu)/pe
        w=Phi*comp/N
        for k in range(N):
            h_all[ebindex(Te[k])]+=w; dir_all[cbindex(cose[k])]+=w; wtot+=w; wE+=w*Te[k]
            if primflag:
                h_prim[ebindex(Te[k])]+=w; dir_prim[cbindex(cose[k])]+=w; wprim+=w; wEp+=w*Te[k]
    # photoelectric
    if p>0:
        w=Phi*p
        i=ebindex(E)
        h_all[i]+=w; dir_all[nb_c-1]+=w; wtot+=w; wE+=w*E
        if primflag:
            h_prim[i]+=w; dir_prim[nb_c-1]+=w; wprim+=w; wEp+=w*E
    # pair
    if q>0 and E>1.022:
        Te=(E-1.022)/2.0
        w=2*Phi*q
        i=ebindex(Te)
        h_all[i]+=w; dir_all[nb_c-1]+=w; wtot+=w; wE+=w*Te
        if primflag:
            h_prim[i]+=w; dir_prim[nb_c-1]+=w; wprim+=w; wEp+=w*Te

for E,F in [(E1,F1),(E2,F2)]:
    add_source(E,F,True)
for E,F in zip(Emid, scat_phi):
    add_source(E,F,False)

print("\nL3 initial spectrum (frac):", h_all/h_all.sum())
print("L3 parent primary:", h_prim/h_prim.sum())
print("L3 mean initial energy all: %.4f MeV, primary: %.4f MeV"%(wE/wtot,wEp/wprim))
print("L3 initial direction:", dir_all/dir_all.sum())
print("L3 initial dir primary:", dir_prim/dir_prim.sum())

# save L3 for later
np.savez('L3.npz', h_all=h_all, h_prim=h_prim, dir_all=dir_all, dir_prim=dir_prim, wE=wE, wtot=wtot, wEp=wEp, wprim=wprim)
