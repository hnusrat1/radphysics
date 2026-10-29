import numpy as np, math
# ---------------- s_w,air calculation ----------------
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
flu_p,flu_s,inter_a,inter_p,inter_s,trans,e_all,e_prim,e_cos,e_sum,e_sum_prim,slab_src,Ntot = A
Ntot=int(Ntot); V_cyl=math.pi*0.2; PHI0=1/10.5**2
ph_edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
Emid=(ph_edges[:-1]+ph_edges[1:])/2
fp = flu_p.sum(1)/Ntot/V_cyl/PHI0; fs = flu_s.sum(1)/Ntot/V_cyl/PHI0
mc2=0.5109989461; d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x=d[:,0]; mu_in=d[:,2]; mu_pe=d[:,3]
def interp(x,xp,yp): return np.interp(x,xp,yp)
def mu_pe_(E): return interp(E,E_x,mu_pe)
s2=np.loadtxt('/tmp/data/estar_water.txt', comments='#'); Ee=s2[:,0]; Scol=s2[:,1]; dd=s2[:,6]
fine=np.arange(0.01,1.401,0.005); df=fine[1]-fine[0]; re2=2.8179403262e-13**2; Ne=3.343e23
qr=np.zeros_like(fine)
for i,E in enumerate(Emid):
    if E<=0.011: continue
    w=fp[i]+fs[i]
    if w<=0: continue
    alpha=E/mc2; Epmax=E/(1+2*alpha); E0s=E-fine
    valid=(E0s>=Epmax)&(E0s<=E)&(fine>0)&(fine<E)
    if not valid.any(): continue
    xg=E/E0s[valid]; cosg=1-(xg-1)/alpha; sin2=1-cosg**2
    dsdc=math.pi*re2*(E0s[valid]/E)**2*((E0s[valid]/E)+(E/E0s[valid])-sin2)
    qr[valid]+=w*Ne*dsdc*(E/(alpha*E0s[valid]**2))
    if mu_pe_(E)>0 and E>0.0005:
        e0=E-0.0005
        if e0>=0.01: qr[np.argmin(np.abs(fine-e0))]+=w*mu_pe_(E)/df
def moller_n(T,W):
    tau=T/mc2; beta2=tau*(tau+2)/(tau+1)**2; Wmax=T/2.0
    out=np.zeros_like(W); v=(W>0.009)&(W<Wmax-1e-9)
    if v.any():
        Wv=W[v]; out[v]=(2.5504e-25*Ne/beta2)*(1/Wv**2)*(1-beta2*Wv/Wmax+(Wv/Wmax)**2/2)
    return out
Phi=np.zeros_like(fine); Qn=0.0
for i in range(len(fine)-1,-1,-1):
    E0=fine[i]; s=0.0
    for j in range(len(fine)-1,i,-1):
        if E0<fine[j]/2 and Phi[j]>0: s+=Phi[j]*moller_n(fine[j],E0)*df
    Q=(Qn if i+1<len(fine) else 0)+(qr[i]+s)*df; Qn=Q
    Phi[i]=Q/interp(fine[i],Ee,Scol)

Kconst=2.5504e-25; New=3.343e23; Nea=3.006e23
Iw=75.0; Ia=85.7
delta_w=interp(fine,Ee,dd)
def hard_loss(T, Ne_g, delta_frac=1.0):
    tau=T/mc2; beta2=tau*(tau+2)/(tau+1)**2; Wmax=T/2.0; Delta=0.01
    if T<=2*Delta: return 0.0
    return (Ne_g*Kconst/beta2)*(math.log(Wmax/Delta) - beta2*(1-Delta/Wmax) + 0.25*(1-(Delta/Wmax)**2))
Scol_w=interp(fine,Ee,Scol)
tau=fine/mc2; beta2=tau*(tau+2)/(tau+1)**2
B_w = Scol_w*beta2/(Kconst*New)
delta_air=delta_w*0.15
B_air = B_w + 2*math.log(Iw/Ia) - delta_air + delta_w
Scol_air = Kconst*Nea*B_air/beta2
# sanity: Scol_air at 1 MeV
print('Scol_air at 1 MeV:', interp(1.0,fine,Scol_air), ' (expect ~1.66)')
print('Scol_air at 0.1 MeV:', interp(0.1,fine,Scol_air), ' water=', interp(0.1,fine,Scol_w))
Hw=np.array([hard_loss(T,New) for T in fine])
Ha=np.array([hard_loss(T,Nea) for T in fine])
Lw=Scol_w-Hw; La=Scol_air-Ha
print('L_delta_w(1 MeV)', interp(1.0,fine,Lw), 'Scol_w', interp(1.0,fine,Scol_w))
print('L_delta_a(1 MeV)', interp(1.0,fine,La))
num=np.trapezoid(Phi*Lw,fine); den=np.trapezoid(Phi*La,fine)
te=Phi[0]*0.01
print('s_w,air (no track-end):', num/den)
print('s_w,air (with track-end):', (num+te)/(den+te))
# also unrestricted ratio
print('unrestricted ratio:', np.trapezoid(Phi*Scol_w,fine)/np.trapezoid(Phi*Scol_air,fine))

# L4 spectrum (25 bins), mean, rad/coll, delta fraction from Phi
Php=np.zeros_like(fine); Qp=0.0
for i in range(len(fine)-1,-1,-1):
    Qp+=qr[i]*df; Php[i]=Qp/interp(fine[i],Ee,Scol)
L4spec=np.zeros(25)
for b in range(25):
    m=(fine>=el_edges[b])&(fine<el_edges[b+1]); L4spec[b]=np.trapezoid(Phi[m],fine[m])
print('L4 spectra:', L4spec)
print('L4 fluence per Phi0:', L4spec.sum())
print('L4 delta fraction:', np.trapezoid(Phi-Php,fine)/np.trapezoid(Phi,fine))
print('L4 mean electron energy:', np.trapezoid(fine*Phi,fine)/np.trapezoid(Phi,fine))
print('L4 rad/coll:', np.trapezoid(Phi*interp(fine,Ee,s2[:,2]),fine)/np.trapezoid(Phi*interp(fine,Ee,Scol),fine))