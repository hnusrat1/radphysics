import numpy as np, pickle

with open('mc_results.pkl','rb') as f:
    d=pickle.load(f)
depth28=d['depth28']; fine5=d['fine5']; score28_at5=d['score28_at5']; depths=d['depths']

def load_xcom():
    with open('/tmp/data/xcom_water.txt') as f:
        lines=[l for l in f if not l.startswith('#') and l.strip()]
    x=np.loadtxt(lines); return x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6]
E_x,mu_coh,mu_incoh,mu_pe,mu_pairn,mu_paire,mu_tot=load_xcom()
def mu_interp(E,col): return np.exp(np.interp(np.log(E),np.log(E_x),np.log(col)))
def mup(E): return dict(compton=mu_interp(E,mu_incoh), pe=mu_interp(E,mu_pe), pair=mu_interp(E,mu_pairn+mu_paire), rayleigh=mu_interp(E,mu_coh), total=mu_interp(E,mu_tot))
def load_estar():
    with open('/tmp/data/estar_water.txt') as f:
        lines=[l for l in f if not l.startswith('#') and l.strip()]
    est=np.loadtxt(lines); return est[:,0],est[:,1],est[:,2],est[:,3],est[:,4],est[:,5],est[:,6]
Ee,Scol,Srad,Stot,csda,yld,den=load_estar()
def interp_e(E,col): return np.exp(np.interp(np.log(E),np.log(Ee),np.log(col)))
def scol_w(E): return interp_e(E,Scol)
def srad_w(E): return interp_e(E,Srad)
def yield_w(E): return interp_e(E,yld)

mc2=0.51099895; re2=7.9408e-26
def kn_dsig(E0,cosT):
    a=E0/mc2; Ep=E0/(1.0+a*(1.0-cosT)); r=Ep/E0
    return 0.5*re2*r**2*(r+1.0/r-(1.0-cosT**2))
def kn_total(E0):
    th=np.linspace(1e-6,np.pi,20000); return np.trapezoid(kn_dsig(E0,np.cos(th))*2*np.pi*np.sin(th),th)
def compton_Ep(E0,cosT):
    a=E0/mc2; return E0/(1.0+a*(1.0-cosT))
def compton_mean_transfer(E0):
    th=np.linspace(1e-6,np.pi,4000); c=np.cos(th)
    w=kn_dsig(E0,c)*2*np.pi*np.sin(th); Tr=E0-compton_Ep(E0,c)
    return np.trapezoid(w*Tr,th)/np.trapezoid(w,th)

# ---- energy-transfer and en coefficients
def mutr_rho(E):
    d=mup(E); c=d['compton']; pe=d['pe']; pair=d['pair']
    Tr_c = (1-compton_mean_transfer(E)/E) if E>0 else 0
    Tr_pair = (1-1.022/E) if E>1.022 else 0
    return c*Tr_c + pe + pair*Tr_pair
def gbar_eff(E):
    # radiative yield weighted over electrons produced at photon E
    d=mup(E); c=d['compton']; pe=d['pe']; pair=d['pair']
    Tr_c=E-compton_mean_transfer(E); Tr_pe=E; Tr_pair=max(E-1.022,0)
    # mean yield for compton recoils
    th=np.linspace(1e-5,np.pi,6000); cc=np.cos(th)
    T=E-compton_Ep(E,cc); dsdo=kn_dsig(E,cc)*2*np.pi*np.sin(th)
    y_T=np.interp(T,Ee,yld,left=0,right=0)
    gc=np.trapezoid(dsdo*y_T*T,th)/max(np.trapezoid(dsdo*T,th),1e-30)
    gp=yield_w(E) if E>0 else 0
    Tp=(E-1.022)/2; gpair=yield_w(Tp) if Tp>0.01 else 0
    num=c*Tr_c*gc + pe*Tr_pe*gp + pair*Tr_pair*gpair
    den=c*Tr_c + pe*Tr_pe + pair*Tr_pair
    return num/max(den,1e-30)
def mken_rho(E):
    tr=mutr_rho(E); return tr*(1-gbar_eff(E))

# test
for E in [1.1732,1.3325,0.5,0.2,0.05]:
    print(E, "mutr/rho",mutr_rho(E),"mken/rho",mken_rho(E),"gbar",gbar_eff(E))

ph_edges=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
mid=(ph_edges[:-1]+ph_edges[1:])/2
conv=1.602e-10  # Gy cm^2 per (MeV/g per Phi0) ... units check below

# Kcol at each depth from depth28
def Kcol_from_fluence28(F_prim28, F_scat28):
    # F arrays fluence per bin per Phi0
    K=0.0; Kcol=0.0
    # primary lines: use exact energies? we only have 28-bin. Use bins 23 and 26 with line energies
    for ib in [23,26]:
        Ew = 1.1732 if ib==23 else 1.3325
        K  += F_prim28[ib]*mutr_rho(Ew)*Ew
        Kcol+= F_prim28[ib]*mken_rho(Ew)*Ew
    for ib in range(28):
        E=mid[ib]
        K += F_scat28[ib]*mutr_rho(E)*E
        Kcol += F_scat28[ib]*mken_rho(E)*E
    return K*conv, Kcol*conv  # Gy cm^2 per Phi0

K_z=np.zeros(len(depths)); Kcol_z=np.zeros(len(depths)); Kcol_prim_z=np.zeros(len(depths)); Kcol_scat_z=np.zeros(len(depths))
for jd in range(len(depths)):
    K,Kcol=Kcol_from_fluence28(depth28[jd,:,0],depth28[jd,:,1])
    K_z[jd]=K; Kcol_z[jd]=Kcol
    # split scattered/primary for dose fraction later
    Kc_p,Kc_s=0.,0.
    for ib in [23,26]:
        Ew=1.1732 if ib==23 else 1.3325
        Kc_p+=depth28[jd,ib,0]*mken_rho(Ew)*Ew*conv
    for ib in range(28):
        Kc_s+=depth28[jd,ib,1]*mken_rho(mid[ib])*mid[ib]*conv
    Kcol_prim_z[jd]=Kc_p; Kcol_scat_z[jd]=Kc_s
print("\ndepth  K/Phi0  Kcol/Phi0 (Gy cm2)")
for jd,zc in enumerate(depths):
    print(zc, K_z[jd]*1e12, Kcol_z[jd]*1e12)
print("\nAt z=5: K/Phi0=",K_z[4]/1e-12 if False else K_z[4],"Kcol=",Kcol_z[4])
print("Kcol_prim(5)/Kcol(5):",Kcol_prim_z[4]/Kcol_z[4])
print("g = 1-Kcol/K at 5:", 1-Kcol_z[4]/K_z[4])

# beta and PDD via electron kernel convolution
# kernel f(u)=6u(R-u)/R^3 on [0,R], R=0.55 cm
R=0.55
u=np.linspace(0,R,401); f=6*u*(R-u)/R**3
# interpolate Kcol_z onto fine z
zfine=np.linspace(0.01,20,4000)
Kcol_fine=np.interp(zfine, depths, Kcol_z, left=0, right=Kcol_z[-1])
Dfine=np.zeros_like(zfine)
for i,z in enumerate(zfine):
    lo=max(0,z-R)
    # Kcol evaluated at z-u, u in [0,min(z,R)]
    umax=min(z,R)
    if umax<=0: continue
    uu=np.linspace(0,umax,201)
    ff=6*uu*(R-uu)/R**3
    Kk=np.interp(z-uu, depths, Kcol_z, left=0)
    Dfine[i]=np.trapezoid(Kk*ff,uu)
imax=np.argmax(Dfine)
print("dmax ~", zfine[imax], "cm; D(dmax)/Phi0=",Dfine[imax])
print("D(5)/Phi0=",Dfine[np.argmin(np.abs(zfine-5.0))])
D5=Dfine[np.argmin(np.abs(zfine-5.0))]
Dmax=Dfine[imax]
print("beta at 5cm = D5/Kcol(5) =", D5/Kcol_z[4])
print("PDD5 = 100*D5/Dmax =", 100*D5/Dmax)
np.save('Kcol_z.npy',Kcol_z); np.save('Dfine.npy',Dfine); np.save('zfine.npy',zfine)