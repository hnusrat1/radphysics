import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
flu_p,flu_s,inter_a,inter_p,inter_s,trans,e_all,e_prim,e_cos,e_sum,e_sum_prim,slab_src,Ntot = A
Ntot=int(Ntot); V_cyl=math.pi*0.2; PHI0=1/10.5**2
ph_edges=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
Emid=(ph_edges[:-1]+ph_edges[1:])/2; Eemid=(el_edges[:-1]+el_edges[1:])/2
fp=flu_p.sum(1)/Ntot/V_cyl/PHI0; fs=flu_s.sum(1)/Ntot/V_cyl/PHI0; ft=fp+fs
print('prim flu', fp.sum(), 'tot', ft.sum(), 'scat frac', fs.sum()/ft.sum())
print('energy scat frac', (Emid*fs).sum()/(Emid*ft).sum())
print('mean photon E', (Emid*ft).sum()/ft.sum())
print('scattered spectrum:', np.round(fs,6))
# L3 means from analytic histograms - recompute quickly from saved e_all/e_prim MC (normalized spectra already have shape)
# Use MC direct for means (well-sampled): mean all=0.36154, prim=0.59776
print('MC L3 means:', e_sum/e_all.sum(), e_sum_prim/e_prim.sum())
espec_all=e_all/e_all.sum(); espec_prim=e_prim/e_prim.sum()
print('L3 spectrum all (MC):', np.round(espec_all,5))
print('L3 spectrum prim (MC):', np.round(espec_prim,5))
print('L3 cos (MC):', np.round(e_cos/e_cos.sum(),5))

# L4: recompute fine integral and bin sum
mc2=0.5109989461; re2=2.8179403262e-13**2; Ne=3.343e23
d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#'); Ex=d[:,0]; mui=d[:,2]; mpe=d[:,3]
s2=np.loadtxt('/tmp/data/estar_water.txt',comments='#'); Ee=s2[:,0]; Scol=s2[:,1]
def interp(x,xp,yp): return np.interp(x,xp,yp)
fine=np.arange(0.01,1.401,0.005); df=fine[1]-fine[0]
qr=np.zeros_like(fine)
for i,E in enumerate(Emid):
    if E<=0.011: continue
    w=ft[i]
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
integ_fine=np.trapezoid(Phi,fine)
L4spec=np.zeros(25)
for b in range(25):
    m=(fine>=el_edges[b])&(fine<el_edges[b+1]); L4spec[b]=np.trapezoid(Phi[m],fine[m])
print('L4 fine integral', integ_fine, ' bin sum', L4spec.sum(), 'diff', integ_fine-L4spec.sum())
# check: maybe fine grid not reaching 1.4 exactly in trapezoid edges
print('fine[0], fine[-1]', fine[0], fine[-1])
# Report L4 using fine integral; spectrum normalized to sum to integral by scaling bins:
print('L4 spectra (per Phi0):', L4spec)
print('scaled to match integral:', L4spec*(integ_fine/L4spec.sum()))