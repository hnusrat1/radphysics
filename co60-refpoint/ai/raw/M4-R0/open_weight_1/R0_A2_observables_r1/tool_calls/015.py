import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
flu_p,flu_s,inter_a,inter_p,inter_s,trans,e_all,e_prim,e_cos,e_sum,e_sum_prim,slab_src,Ntot = A
V_cyl=math.pi*0.2; PHI0=1/10.5**2
ph_edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
cos_edges=np.linspace(-1,1,11)
Emid=(ph_edges[:-1]+ph_edges[1:])/2
Eemid=(el_edges[:-1]+el_edges[1:])/2
fp = flu_p.sum(1)/Ntot/V_cyl/PHI0
fs = flu_s.sum(1)/Ntot/V_cyl/PHI0
ftot=(fp+fs)
s = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=s[:,0]; Scol=s[:,1]; Srad=s[:,2]; Rcsda=s[:,4]; Y=s[:,5]; dd=s[:,6]
def interp(x,xp,yp): return np.interp(x,xp,yp)
def S_c(E): return interp(E,Ee,Scol)
def S_r(E): return interp(E,Ee,Srad)
def Rng(E): return interp(E,Ee,Rcsda)*1.0  # g/cm2 = cm at rho=1
def Yld(E): return interp(E,Ee,Y)

mc2=0.5109989461
d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x=d[:,0]; mu_co=d[:,1]; mu_in=d[:,2]; mu_pe=d[:,3]; mu_pn=d[:,4]; mu_pef=d[:,5]; mu_tot=d[:,6]
def mut(E): return interp(E,E_x,mu_tot)
def muc_(E): return interp(E,E_x,mu_co)
def mui_(E): return interp(E,E_x,mu_in)
def mup_(E): return interp(E,E_x,mu_pe)
def mupair(E): return interp(E,E_x,mu_pn)+interp(E,E_x,mu_pef)

# ---------- mu_tr, mu_en per energy ----------
def mu_tr(E):
    alpha=E/mc2
    cos=np.linspace(-1,1,6001)
    f=KNf(alpha,cos)
    x=1+alpha*(1-cos); Ep=E/x; T=E-Ep
    meanT=np.trapezoid(T*f,cos)/np.trapezoid(f,cos)
    return mui_(E)*meanT/E + mup_(E)*(E-0.0005)/E + mupair(E)*(E-1.022)/E
def KNf(alpha,cos):
    x=1+alpha*(1-cos); r=1/x; return r*r*(r+x-(1-cos*cos))

# K and Kcol (we need mu_en = mu_tr*(1-g_e)); first compute g_e via electron yield sampling
# g_e(E) = <T*Y(T)>/<T> for electrons released by photons of energy E
def g_photon(E):
    alpha=E/mc2
    cos=np.linspace(-1,1,6001)
    f=KNf(alpha,cos); x=1+alpha*(1-cos); Ep=E/x; T=E-Ep
    # weight by dsigma/dcos ~ f; yield Y(T)
    num=np.trapezoid(T*Yld(T)*f,cos); den=np.trapezoid(T*f,cos)
    # include photoelectric and pair (negligible)
    return num/den

K=0.0; Kc=0.0; Kc_prim=0.0; Kc_scat=0.0
tr_arr=np.zeros_like(Emid); en_arr=np.zeros_like(Emid)
for i,E in enumerate(Emid):
    if E<=0.01: continue
    mtr=mu_tr(E); g=g_photon(E); men=mtr*(1-g)
    tr_arr[i]=mtr; en_arr[i]=men
    K   += ftot[i]*E*mtr
    Kc  += ftot[i]*E*men
    Kc_prim += fp[i]*E*men
    Kc_scat += fs[i]*E*men
Gy=1.602176634e-10
print('K/Phi0 (Gy cm2)', K*Gy)
print('Kcol/Phi0 (Gy cm2)', Kc*Gy)
print('Kcol_prim/Phi0', Kc_prim*Gy, 'Kcol_scat/Phi0', Kc_scat*Gy)
print('g = 1-Kcol/K', 1-Kc/K)
print('dose fraction parent scattered (Kcol_scat/Kcol):', Kc_scat/Kc)

# ---------- electron initial source q̃(E0) fine from photon spectra ----------
fine=np.arange(0.01,1.401,0.005); df=fine[1]-fine[0]
qr=np.zeros_like(fine)
# Compton contribution: for each photon bin, exact dsigma/dE0 on fine grid
re2=2.8179403262e-13**2
Ne=3.343e23
for i,E in enumerate(Emid):
    if E<=0.011: continue
    w=ftot[i]*mui_(E)  # cm^-1 per Phi0 (per bin)
    if w<=0: continue
    alpha=E/mc2
    # E0 range: 0 .. Tmax=E*2alpha/(1+2alpha)
    # compute dσ/dE0 via sampling (histogram) - faster and accurate enough
    # Instead analytic: use grid
    x=1+2*alpha
    # E' from E to E' min
    # For each fine E0: E' = E-E0, x0=E/E'
    Emax_e = E*(2*alpha)/(1+2*alpha)
    Epmax=E/(1+2*alpha)
    E0s=E-fine  # E'
    valid = (E0s>=Epmax)&(E0s<=E)&(fine>0)
    # need fine<E (E0>0)
    xg = E/E0s
    cosg = 1-(xg-1)/alpha
    sin2=1-cosg**2
    dsdc = math.pi*re2*(E0s/E)**2*((E0s/E)+(E/E0s)-sin2)
    dsdE0 = dsdc*(E/(alpha*E0s**2))
    valid &= np.isfinite(dsdE0)
    qr[valid]+= w*Ne*dsdE0[valid]
    # photoelectric delta
    if mup_(E)>0 and E>0.0005:
        e0=E-0.0005
        if e0>=0.01:
            ib=np.argmin(np.abs(fine-e0))
            qr[ib]+= ftot[i]*mup_(E)/df
    if E>1.022 and mupair(E)>0:
        e0=(E-1.022)/2
        if e0>=0.01:
            ib=np.argmin(np.abs(fine-e0))
            qr[ib]+= 2*ftot[i]*mupair(E)/df
print('q̃ integral (cm^-1 per Phi0):', np.trapezoid(qr,fine))
print('total initial energy per cm per Phi0 (MeV/cm):', np.trapezoid(fine*qr,fine))

# ---------- CSDA + delta solve ----------
def moller_n(T,W):
    # per cm per MeV (units cm^-1 MeV^-1): Ne * dσ/dW
    tau=T/mc2; beta2=tau*(tau+2)/(tau+1)**2
    Wmax=T/2.0
    out=np.zeros_like(W)
    v=(W>0.009)&(W<Wmax-1e-9)
    if v.any():
        Wv=W[v]
        out[v]=(2.5504e-25*Ne/beta2)*(1.0/Wv**2)*(1-beta2*Wv/Wmax+(Wv/Wmax)**2/2.0)
    return out

Phi=np.zeros_like(fine); Q=np.zeros_like(fine)
# top-down
for i in range(len(fine)-1,-1,-1):
    E0=fine[i]
    s=0.0
    for j in range(len(fine)-1,i,-1):
        if E0 < fine[j]/2 and Phi[j]>0:
            s += Phi[j]*moller_n(fine[j],E0)*df
    Q[i] = (Q[i+1] if i+1<len(fine) else 0.0) + (qr[i]+s)*df
    Phi[i] = Q[i]/S_c(E0)
print('electron fluence integral /Phi0 (>=10keV):', np.trapezoid(Phi,fine))
# primary-only fluence (no delta)
Php=np.zeros_like(fine); Qp=0.0
for i in range(len(fine)-1,-1,-1):
    Qp += qr[i]*df
    Php[i]=Qp/S_c(fine[i])
print('primary-track electron fluence integral /Phi0:', np.trapezoid(Php,fine))
print('delta fluence integral /Phi0:', np.trapezoid(Phi-Php,fine))
print('delta fraction:', np.trapezoid(Phi-Php,fine)/np.trapezoid(Phi,fine))
print('mean electron energy (all):', np.trapezoid(fine*Phi,fine)/np.trapezoid(Phi,fine))

# Layer4 spectrum on 25 bins
L4spec=np.zeros(NB_EL:=25)
for b in range(25):
    lo,hi=el_edges[b],el_edges[b+1]
    m=(fine>=lo)&(fine<hi)
    L4spec[b]=np.trapezoid(Phi[m],fine[m])
print('L4 spectrum (per Phi0 per bin):', L4spec)
print('L4 fluence per Phi0 (sum):', L4spec.sum())
print('radiative/collision:', np.trapezoid(Phi*S_r(fine),fine)/np.trapezoid(Phi*S_c(fine),fine))
# D from electron fluence
D_e = np.trapezoid(Phi*S_c(fine),fine)*Gy
print('D/Phi0 from electron fluence (Gy cm2):', D_e)