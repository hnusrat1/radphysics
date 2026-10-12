import numpy as np, math
# quick sensitivity of s_w,air to delta_air
# reuse precomputed Phi? recompute quickly is heavy; instead just approximate ratio using Scol_w and Phi shape from saved values
# We'll recompute fully but compactly (skip photon, reuse fine arrays built before by re-running)
mc2=0.5109989461; Kconst=2.5504e-25; New=3.343e23; Nea=3.006e23; Iw=75.0; Ia=85.7
s2=np.loadtxt('/tmp/data/estar_water.txt', comments='#'); Ee=s2[:,0]; Scol=s2[:,1]; dd=s2[:,6]
fine=np.arange(0.01,1.401,0.005); df=fine[1]-fine[0]
# use a representative Phi (normalized) -- approximate with the one we computed: rough shape Phi(E)=Q0/S where Q0 decays; 
# Just load the fine arrays from previous computation is not saved; re-run compact via saved mc file is heavy but fine
A=np.load('/tmp/mc_tallies_5e6.npy',allow_pickle=True); flu_p,flu_s=A[0],A[1]; Ntot=int(A[12]); V=math.pi*0.2; P=1/10.5**2
ph_edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
Emid=(ph_edges[:-1]+ph_edges[1:])/2
fp=flu_p.sum(1)/Ntot/V/P; fs=flu_s.sum(1)/Ntot/V/P
d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#'); Ex=d[:,0]; mui=d[:,2]; mpe=d[:,3]
def interp(x,xp,yp): return np.interp(x,xp,yp)
re2=2.8179403262e-13**2; Ne=3.343e23
qr=np.zeros_like(fine)
for i,E in enumerate(Emid):
    if E<=0.011: continue
    w=fp[i]+fs[i]
    alpha=E/mc2; Epmax=E/(1+2*alpha); E0s=E-fine
    valid=(E0s>=Epmax)&(E0s<=E)&(fine>0)&(fine<E)
    if not valid.any(): continue
    xg=E/E0s[valid]; cosg=1-(xg-1)/alpha; sin2=1-cosg**2
    dsdc=math.pi*re2*(E0s[valid]/E)**2*((E0s[valid]/E)+(E/E0s[valid])-sin2)
    qr[valid]+=w*Ne*dsdc*(E/(alpha*E0s[valid]**2))
    if E>0.0005:
        e0=E-0.0005
        if e0>=0.01: qr[np.argmin(np.abs(fine-e0))]+=w*interp(E,Ex,mpe)/df
def moller_n(T,W):
    tau=T/mc2; b2=tau*(tau+2)/(tau+1)**2; Wm=T/2; out=np.zeros_like(W); v=(W>0.009)&(W<Wm-1e-9)
    if v.any():
        Wv=W[v]; out[v]=(2.5504e-25*Ne/b2)*(1/Wv**2)*(1-b2*Wv/Wm+(Wv/Wm)**2/2)
    return out
Phi=np.zeros_like(fine); Qn=0
for i in range(len(fine)-1,-1,-1):
    E0=fine[i]; s=0.0
    for j in range(len(fine)-1,i,-1):
        if E0<fine[j]/2 and Phi[j]>0: s+=Phi[j]*moller_n(fine[j],E0)*df
    Q=(Qn if i+1<len(fine) else 0)+(qr[i]+s)*df; Qn=Q; Phi[i]=Q/interp(fine[i],Ee,Scol)
dw=interp(fine,Ee,dd)
tau=fine/mc2; b2=tau*(tau+2)/(tau+1)**2
Scol_w=interp(fine,Ee,Scol); Bw=Scol_w*b2/(Kconst*New)
def hard(T,neg):
    tau2=T/mc2; b=tau2*(tau2+2)/(tau2+1)**2; Wm=T/2; Del=0.01
    if T<=2*Del: return 0.0
    return (neg*Kconst/b)*(math.log(Wm/Del)-b*(1-Del/Wm)+0.25*(1-(Del/Wm)**2))
Hw=np.array([hard(T,New) for T in fine]); Lw=Scol_w-Hw
for fac in [0.0,0.05,0.1,0.15,0.2,0.3]:
    da=dw*fac
    Ba=Bw+2*math.log(Iw/Ia)-da+dw
    Sa=Kconst*Nea*Ba/b2; La=np.array([hard(T,Nea) for T in fine]); La=Sa-La
    s=np.trapezoid(Phi*Lw,fine)/np.trapezoid(Phi*La,fine)
    print('delta_air factor',fac,'s_w,air',round(s,4),' Scol_air(1MeV)',round(interp(1,fine,Sa),4))