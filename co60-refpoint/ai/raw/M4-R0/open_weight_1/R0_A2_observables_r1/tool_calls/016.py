import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
flu_p,flu_s,inter_a,inter_p,inter_s,trans,e_all,e_prim,e_cos,e_sum,e_sum_prim,slab_src,Ntot = A
V_cyl=math.pi*0.2; PHI0=1/10.5**2
ph_edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
Emid=(ph_edges[:-1]+ph_edges[1:])/2
fp = flu_p.sum(1)/Ntot/V_cyl/PHI0
fs = flu_s.sum(1)/Ntot/V_cyl/PHI0
ftot=fp+fs
s = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=s[:,0]; Scol=s[:,1]; Srad=s[:,2]; Rcsda=s[:,4]; Y=s[:,5]; dd=s[:,6]
def interp(x,xp,yp): return np.interp(x,xp,yp)
def S_c(E): return interp(E,Ee,Scol)
def S_r(E): return interp(E,Ee,Srad)
def Yld(E): return interp(E,Ee,Y)
mc2=0.5109989461
d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x=d[:,0]; mu_co=d[:,1]; mu_in=d[:,2]; mu_pe=d[:,3]; mu_pn=d[:,4]; mu_pef=d[:,5]; mu_tot=d[:,6]
def muc_(E): return interp(E,E_x,mu_co)
def mui_(E): return interp(E,E_x,mu_in)
def mup_(E): return interp(E,E_x,mu_pe)
def mupair(E): return interp(E,E_x,mu_pn)+interp(E,E_x,mu_pef)
def KNf(alpha,cos):
    x=1+alpha*(1-cos); r=1/x; return r*r*(r+x-(1-cos*cos))
def mu_tr(E):
    alpha=E/mc2; cos=np.linspace(-1,1,4001); f=KNf(alpha,cos)
    x=1+alpha*(1-cos); Ep=E/x; T=E-Ep
    meanT=np.trapezoid(T*f,cos)/np.trapezoid(f,cos)
    return mui_(E)*meanT/E + mup_(E)*(E-0.0005)/E + mupair(E)*(E-1.022)/E
def g_photon(E):
    alpha=E/mc2; cos=np.linspace(-1,1,4001); f=KNf(alpha,cos); x=1+alpha*(1-cos); Ep=E/x; T=E-Ep
    return np.trapezoid(T*Yld(T)*f,cos)/np.trapezoid(T*f,cos)
Gy=1.602176634e-10
K=0.0; Kc=0.0; Kc_prim=0.0; Kc_scat=0.0
for i,E in enumerate(Emid):
    if E<=0.011: continue
    mtr=mu_tr(E); men=mtr*(1-g_photon(E))
    K+=ftot[i]*E*mtr; Kc+=ftot[i]*E*men; Kc_prim+=fp[i]*E*men; Kc_scat+=fs[i]*E*men
print('K/Phi0',K*Gy,'Kcol/Phi0',Kc*Gy,'g',1-Kc/K,'dose frac scat',Kc_scat/Kc)
print('Kcol_prim',Kc_prim*Gy,'Kcol_scat',Kc_scat*Gy)

# q̃ (fixed)
fine=np.arange(0.01,1.401,0.005); df=fine[1]-fine[0]
re2=2.8179403262e-13**2; Ne=3.343e23
qr=np.zeros_like(fine)
for i,E in enumerate(Emid):
    if E<=0.011: continue
    w=ftot[i]
    if w<=0: continue
    alpha=E/mc2
    Epmax=E/(1+2*alpha)
    E0s=E-fine
    valid=(E0s>=Epmax)&(E0s<=E)&(fine>0.0)&(fine<E)
    if not valid.any(): continue
    xg=E/E0s[valid]
    cosg=1-(xg-1)/alpha
    sin2=1-cosg**2
    dsdc=math.pi*re2*(E0s[valid]/E)**2*((E0s[valid]/E)+(E/E0s[valid])-sin2)
    dsdE0=dsdc*(E/(alpha*E0s[valid]**2))
    qr[valid]+= w*Ne*dsdE0
    if mup_(E)>0 and E>0.0005:
        e0=E-0.0005
        if e0>=0.01:
            qr[np.argmin(np.abs(fine-e0))]+= w*mup_(E)/df
    if E>1.022 and mupair(E)>0:
        e0=(E-1.022)/2
        if e0>=0.01:
            qr[np.argmin(np.abs(fine-e0))]+= 2*w*mupair(E)/df
print('q̃ integral',np.trapezoid(qr,fine),' ∫E q̃',np.trapezoid(fine*qr,fine),' expect K/gy~',K/Gy)

def moller_n(T,W):
    tau=T/mc2; beta2=tau*(tau+2)/(tau+1)**2; Wmax=T/2.0
    out=np.zeros_like(W); v=(W>0.009)&(W<Wmax-1e-9)
    if v.any():
        Wv=W[v]; out[v]=(2.5504e-25*Ne/beta2)*(1.0/Wv**2)*(1-beta2*Wv/Wmax+(Wv/Wmax)**2/2.0)
    return out
Phi=np.zeros_like(fine)
for i in range(len(fine)-1,-1,-1):
    E0=fine[i]; s=0.0
    for j in range(len(fine)-1,i,-1):
        if E0<fine[j]/2 and Phi[j]>0: s+=Phi[j]*moller_n(fine[j],E0)*df
    Q=(Qn if i+1<len(fine) else 0.0)+(qr[i]+s)*df; Qn=Q
    Phi[i]=Q/S_c(E0)
Php=np.zeros_like(fine); Qp=0.0
for i in range(len(fine)-1,-1,-1):
    Qp+=qr[i]*df; Php[i]=Qp/S_c(fine[i])
Phi_tot_integ=np.trapezoid(Phi,fine)
print('L4 fluence/Phi0:',Phi_tot_integ)
print('delta fract:',np.trapezoid(Phi-Php,fine)/Phi_tot_integ)
print('mean e energy:',np.trapezoid(fine*Phi,fine)/Phi_tot_integ)
L4spec=np.zeros(25)
for b in range(25):
    m=(fine>=el_edges[b])&(fine<el_edges[b+1]); L4spec[b]=np.trapezoid(Phi[m],fine[m])
print('L4 spec sum',L4spec.sum())
print('L4spec:',L4spec)
print('rad/coll:',np.trapezoid(Phi*S_r(fine),fine)/np.trapezoid(Phi*S_c(fine),fine))
D_e=np.trapezoid(Phi*S_c(fine),fine)*Gy
print('D/Phi0 from electrons:',D_e, ' Kcol:',Kc*Gy,' beta:',D_e/(Kc*Gy))