import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
me=0.51099895
def trapz(y,x): return np.trapezoid(y,x)

d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
_Em=d[:,0]; _incoh=d[:,2]; _pe=d[:,3]; _pair=d[:,4]+d[:,5]
f_incoh=interp1d(_Em,_incoh,bounds_error=False,fill_value=(_incoh[0],_incoh[-1]))
f_pe=interp1d(_Em,_pe,bounds_error=False,fill_value=(_pe[0],_pe[-1]))
f_pair=interp1d(_Em,_pair,bounds_error=False,fill_value=(_pair[0],_pair[-1]))
d2=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
_Ee=d2[:,0]; _Sc=d2[:,1]; _Sr=d2[:,2]; _St=d2[:,3]; _Y=d2[:,5]
f_Sc=interp1d(_Ee,_Sc,bounds_error=False,fill_value=(_Sc[0],_Sc[-1]))
f_Sr=interp1d(_Ee,_Sr,bounds_error=False,fill_value=(_Sr[0],_Sr[-1]))
f_Y=interp1d(_Ee,_Y,bounds_error=False,fill_value=(_Y[0],_Y[-1]))
PH_EDGES=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
                   0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
phc=(PH_EDGES[:-1]+PH_EDGES[1:])/2
E1=1.1732; E2=1.3325
def kn_ratio(E,mu):
    a=E/me; r=1/(1+a*(1-mu)); return r*r*(r+1/r-(1-mu**2))
def mu_en(E):
    a=E/me
    num,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu)*(E-E/(1+a*(1-mu))),-1,1)
    den,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu),-1,1)
    TeC=num/den if den>0 else 0; fC=TeC/E
    mt=f_incoh(E)*fC+f_pe(E)+(f_pair(E)*(1-1.022/E) if E>1.022 else 0)
    gC=f_Y(max(TeC,0.011)); gPE=f_Y(E); gpair=f_Y(max((E-1.022)/2,0.011)) if E>1.022 else 0
    gbar=(f_incoh(E)*fC*gC+f_pe(E)*gPE+(f_pair(E)*(1-1.022/E)*gpair if E>1.022 else 0))/max(mt,1e-30)
    return mt*(1-gbar)
def mu_tr(E):
    a=E/me
    num,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu)*(E-E/(1+a*(1-mu))),-1,1)
    den,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu),-1,1)
    fC=(num/den/E) if den>0 else 0
    return f_incoh(E)*fC+f_pe(E)+(f_pair(E)*(1-1.022/E) if E>1.022 else 0)

r=np.load('mc_results.npz',allow_pickle=True)
fl=np.load('fluence_results.npz',allow_pickle=True)
EG=fl['EG']; Phi_all=fl['Phi_all']; Phi_prim=fl['Phi_prim']; Phi_scat=fl['Phi_scat']; Phi_p0=fl['Phi_p0']
Vcyl=np.pi*0.2; N=int(r['N']); i5=np.argmin(abs(r['depths']-5.0))
prim_phi=0.5*(np.exp(-0.0653626*5)+np.exp(-0.06122524*5))
k=prim_phi/(r['prim_tl'][i5]/Vcyl/N)
spec_scat_per=r['spec_scat']/Vcyl/N*k
f1=0.5*np.exp(-0.0653626*5); f2=0.5*np.exp(-0.06122524*5)
C=1.602e-10

# Kcol and K per Phi0
Kcol_p=(f1*mu_en(E1)*E1 + f2*mu_en(E2)*E2)*C
K_p=(f1*mu_tr(E1)*E1 + f2*mu_tr(E2)*E2)*C
Kcol_s=sum(spec_scat_per[i]*mu_en(phc[i])*phc[i] for i in range(28))*C
K_s=sum(spec_scat_per[i]*mu_tr(phc[i])*phc[i] for i in range(28))*C
Kcol=(Kcol_p+Kcol_s); K=K_p+K_s
print("Kcol_p,Kcol_s,Kcol:",Kcol_p,Kcol_s,Kcol)
print("K_p,K_s,K:",K_p,K_s,K)

# D from electron fluence
Sc=f_Sc(np.maximum(EG,0.011)); Sr=f_Sr(np.maximum(EG,0.011))
D= C*trapz(Phi_all*Sc,EG)
D_prim= C*trapz(Phi_prim*Sc,EG)
D_scat= C*trapz(Phi_scat*Sc,EG)
print("D =",D," D_prim-parent:",D_prim," D_scat-parent:",D_scat," sum:",D_prim+D_scat)
print("beta=D/Kcol:",D/Kcol)
print("g=1-Kcol/K:",1-Kcol/K)
print("dose fraction parent scattered:",D_scat/(D_prim+D_scat))

# check L4 spectrum zeros
EL_EDGES=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
                   0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
def fluence_bins(EG,Phi,edges):
    out=np.zeros(len(edges)-1)
    for i in range(len(edges)-1):
        m=(EG>=edges[i])&(EG<edges[i+1])
        out[i]=trapz(Phi[m],EG[m]) if m.sum()>1 else (Phi[m][0]*(EG[m][0] if m.sum()==1 else 0))
    return out
L4=fluence_bins(EG,Phi_all,EL_EDGES)
print("L4 fluence spectrum:",np.array2string(L4,precision=6))
print("total:",L4.sum())
# electron fluence per Phi0 and delta fraction
print("flu_per_phi0:",L4.sum())
print("delta fraction:",1-trapz(Phi_p0,EG)/trapz(Phi_all,EG))