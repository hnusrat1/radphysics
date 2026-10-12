import numpy as np, math

d=np.load('mc_results.npz'); L3=np.load('L3.npz')
h_all=L3['h_all']; h_prim=L3['h_prim']
Kcol_Phi=5.352357476400485e-12
Emean_all=float(L3['wE']/L3['wtot'])
print("mean initial energy all",Emean_all)

edges_e=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
nb_e=25

# fine grid
Egrid=np.arange(0.010,1.4005,0.002)
nE=len(Egrid)
# piecewise p (frac per MeV) from L3 bins
def p_of_E(E):
    i=int(np.searchsorted(edges_e,E,side='right')-1)
    i=max(0,min(nb_e-1,i))
    w=edges_e[i+1]-edges_e[i]
    return h_all[i]/w
p=np.array([p_of_E(E) for E in Egrid])
# normalize check
print("integral p dE", np.trapezoid(p,Egrid))

# stopping power water and air tables
Ew=np.array([0.01,0.015,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.5])
Sw=np.array([22.36,16.10,12.98,9.00,5.93,4.72,4.10,3.35,3.05,2.63,2.39,2.23,2.12,2.04,1.98,1.93,1.89,1.86,1.84,1.83,1.84],float)
Sa=np.array([20.68,14.94,12.08,8.60,5.51,4.37,3.78,3.10,2.81,2.40,2.16,2.01,1.91,1.84,1.78,1.73,1.69,1.66,1.64,1.62,1.62],float)
def Swater(E): return float(np.exp(np.interp(np.log(E),np.log(Ew),np.log(Sw))))
def Sair(E): return float(np.exp(np.interp(np.log(E),np.log(Ew),np.log(Sa))))
S_col=np.array([Swater(E) for E in Egrid])
S_air=np.array([Sair(E) for E in Egrid])

# q(E0) per Phi0
K_vol_phi = Kcol_Phi*6.242e9   # MeV/cm3 per Phi0
q = K_vol_phi*p/Emean_all       # per cm3 MeV per Phi0
# cumulative integral from high to low
dE=Egrid[1]-Egrid[0]
I=np.zeros(nE)
acc=0.0
for i in range(nE-1,-1,-1):
    acc += q[i]*dE
    I[i]=acc
Phi_p = I/S_col   # fluence per MeV per Phi0
print("primary electron fluence per Phi0 (>=10keV):", np.trapezoid(Phi_p,Egrid))

# Moller cross section dsigma/dT (cm^2/MeV) for incident kinetic E (MeV), recoil T (MeV)
r0=2.8179403262e-13  # cm
mc2=0.5109989461
twopi_r02=2*np.pi*r0*r0
def moller(E,T):
    if T<=0 or T>=E/2: return 0.0
    gamma=1.0+E/mc2
    beta2=1.0-1.0/(gamma*gamma)
    A=twopi_r02*mc2/beta2
    term= 1.0/(T*T) + 1.0/( (E-T)*(E-T) ) - (2*gamma-1)/(gamma*gamma*T*(E-T)) + ((gamma-1)**2)/(gamma*gamma*mc2*mc2)
    return A*term

def delta_spectrum(Phi_in):
    qd=np.zeros(nE)
    for i in range(nE):
        Eprim=Egrid[i]
        if Eprim<0.021: continue
        Ph=Phi_in[i]
        if Ph<=0: continue
        Tmax=Eprim/2.0
        # loop T grid up to Tmax
        # vectorize
        idx=np.arange(i, nE)  # T energy >= Egrid[i] (T starts near Egrid[i])
        T=Egrid[idx]
        T=T[T<=Tmax]
        if len(T)==0: continue
        ds=np.array([moller(Eprim,t) for t in T])
        qd[idx[:len(T)]] += Ph*3.343e23*ds*dE
    # integrate qd from E to inf
    Id=np.zeros(nE)
    acc=0.0
    for i in range(nE-1,-1,-1):
        acc += qd[i]*dE
        Id[i]=acc
    return Id/S_col

print("computing delta gen1...")
Phi_d1=delta_spectrum(Phi_p)
Phi_tot=Phi_p+Phi_d1
print("delta1 fluence",np.trapezoid(Phi_d1,Egrid), "fraction of primary", np.trapezoid(Phi_d1,Egrid)/np.trapezoid(Phi_p,Egrid))
print("computing delta gen2...")
Phi_d2=delta_spectrum(Phi_d1)
Phi_tot=Phi_p+Phi_d1+Phi_d2
print("delta2 fluence",np.trapezoid(Phi_d2,Egrid))

# bin into 25 bins
flux_bin=np.zeros(nb_e)
for i in range(nb_e):
    lo,hi=edges_e[i],edges_e[i+1]
    m=(Egrid>=lo)&(Egrid<hi)
    flux_bin[i]=np.trapezoid(Phi_tot[m],Egrid[m]) if m.sum()>1 else 0.0
print("\nL4_electron_fluence_spectrum (per Phi0):", flux_bin)
print("L4 total fluence per Phi0:", flux_bin.sum())
delfrac=(np.trapezoid(Phi_d1+Phi_d2,Egrid))/np.trapezoid(Phi_tot,Egrid)
print("L4 delta fraction:", delfrac)
meanE=np.trapezoid(Egrid*Phi_tot,Egrid)/np.trapezoid(Phi_tot,Egrid)
print("L4 mean electron energy MeV:", meanE)

# dose check
Dcheck=1.602e-10*np.trapezoid(Phi_tot*S_col,Egrid)
print("D from electron spectrum Gy cm2 (should equal Kcol):", Dcheck, "Kcol:", Kcol_Phi)

# radiative to collision
Srad=0.006*np.array([E**1.7 for E in Egrid])
ratio_rc=np.trapezoid(Phi_tot*Srad,Egrid)/np.trapezoid(Phi_tot*S_col,Egrid)
print("L4 radiative_to_collision:", ratio_rc)

# Spencer-Attix water/air ratio with track ends Delta=10 keV
Delta=0.01
m=Egrid>=Delta
num=np.trapezoid(Phi_tot[m]*S_col[m],Egrid[m])
den=np.trapezoid(Phi_tot[m]*S_air[m],Egrid[m])
# track-end term: fluence at Delta (per MeV) times Delta
# get Phi at Delta ~ first few points
PhiD=np.interp(Delta,Egrid,Phi_tot)
saw=(num+PhiD*Delta)/(den+PhiD*Delta)
print("s_w_air (SA, track end):",saw)

# g: radiative yield from initial electron distribution
def g_rad(E0):
    eg=Egrid[Egrid<=E0]
    if len(eg)<2: return 0.0
    Srad_e=0.006*np.array([e**1.7 for e in eg])
    Sc_e=np.array([Swater(e) for e in eg])
    # integral of Srad/Sc from 0.01 to E0
    val=np.trapezoid(np.array([ (0.006*e**1.7)/max(Swater(e),1e-6) for e in eg]),eg)
    return val/E0
# weighted by q(E0)*E0
g_num=0.0; g_den=0.0
for i in range(nE):
    E0=Egrid[i]
    g_num += q[i]*E0*g_rad(E0)*dE
    g_den += q[i]*E0*dE
g=g_num/g_den
print("L5 g (radiative yield):",g)
print("K=Kcol/(1-g):",Kcol_Phi/(1-g))

# L4 angular: model with energy dependent mean cos
def mean_cos(E):
    return min(0.75,0.15+0.5*E/1.2)
def ang_frac(c):
    # f(mu) prop exp(a mu), get a by bisection so mean = c
    def meanof(a):
        if abs(a)<1e-8: return 0.0
        return 1.0/np.tanh(a)-1.0/a
    lo,hi=0.0,8.0
    if c<=0: return np.full(10,0.1)
    if meanof(8)<c: a=8.0
    else:
        for _ in range(60):
            mid=(lo+hi)/2
            if meanof(mid)<c: lo=mid
            else: hi=mid
        a=lo
    edges=np.linspace(-1,1,11)
    # integral exp(a mu)
    vals=np.array([(np.exp(a*edges[k+1])-np.exp(a*edges[k]))/(a) for k in range(10)]) if a>1e-8 else np.full(10,0.1)
    return vals/vals.sum()
ang=np.zeros(10)
totw=0.0
for i in range(nE):
    fr=ang_frac(min(0.78,max(0.0,mean_cos(Egrid[i]))))
    ang+=fr*Phi_tot[i]*dE
    totw+=Phi_tot[i]*dE
ang/=ang.sum()
print("L4_angular (model):",ang,"sum",ang.sum())

np.savez('L4.npz', Egrid=Egrid, Phi_tot=Phi_tot, Phi_p=Phi_p, Phi_d1=Phi_d1, Phi_d2=Phi_d2,
         flux_bin=flux_bin, ang=ang, ratio_rc=ratio_rc, g=g, saw=saw)
