import numpy as np
from scipy.interpolate import interp1d
me=0.51099895
r0=2.8179403e-13; NA=6.02214076e23; Ne_w=NA*(10.0/18.015268)
d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
_Em=d[:,0]; _incoh=d[:,2]; _pe=d[:,3]; _pair=d[:,4]+d[:,5]
f_incoh=interp1d(_Em,_incoh,bounds_error=False,fill_value=(_incoh[0],_incoh[-1]))
f_pe=interp1d(_Em,_pe,bounds_error=False,fill_value=(_pe[0],_pe[-1]))
f_pair=interp1d(_Em,_pair,bounds_error=False,fill_value=(_pair[0],_pair[-1]))
d2=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
_Ee=d2[:,0]; _Sc=d2[:,1]; _Sr=d2[:,2]; _St=d2[:,3]
f_Sc=interp1d(_Ee,_Sc,bounds_error=False,fill_value=(_Sc[0],_Sc[-1]))
f_Sr=interp1d(_Ee,_Sr,bounds_error=False,fill_value=(_Sr[0],_Sr[-1]))
f_St=interp1d(_Ee,_St,bounds_error=False,fill_value=(_St[0],_St[-1]))
EL_EDGES=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
                   0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
PH_EDGES=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
                   0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
phc=(PH_EDGES[:-1]+PH_EDGES[1:])/2
E1=1.1732; E2=1.3325

def kn_ratio(E,mu):
    a=E/me; r=1/(1+a*(1-mu)); return r*r*(r+1/r-(1-mu**2))
def compton_pdf_grid(E, EG):
    a=E/me
    mu=np.linspace(-1+1e-9,1-1e-9,6000)
    r=1/(1+a*(1-mu)); Ep=E*r; Ee=E-Ep
    dmu=mu[1]-mu[0]
    w=2*np.pi*kn_ratio(E,mu)*dmu
    dP=w; tot=dP.sum()
    edges=np.append(EG, EG[-1]+0.05)
    pdf_hist,_=np.histogram(Ee,bins=edges,weights=dP/tot)
    widths=np.diff(edges)
    return pdf_hist/widths

def moller_dsdT(Ep,T):
    gamma=(Ep+me)/me; b2=1-1/gamma**2
    pref=2*np.pi*r0**2*me/b2
    term=1/T**2+1/(Ep-T)**2-(2*gamma-1)/gamma**2/(T*(Ep-T))+((gamma-1)/gamma)**2/me**2
    return pref*term

def solve_fluence(S0, EG, iters=60):
    N=len(EG); dE=np.gradient(EG)
    St=f_St(EG)
    def cumint(f):
        vals=f*dE; c=np.zeros(N); c[-1]=vals[-1]/2
        for i in range(N-2,-1,-1): c[i]=c[i+1]+(vals[i]+vals[i+1])/2
        return c
    K=np.zeros((N,N))
    for j in range(N):
        limit=EG[j]/2; iend=np.searchsorted(EG,limit,side='right')
        if iend>0: K[j,:iend]=Ne_w*moller_dsdT(EG[j],EG[:iend])
    I0=cumint(S0)
    Phi=np.zeros(N)
    for it in range(iters):
        Sd=K.T.dot(Phi*dE)
        Id=cumint(Sd)
        Phi=np.maximum((I0+Id)/St,0)
    Phi_p0=I0/St
    return Phi, Phi_p0

r=np.load('mc_results.npz',allow_pickle=True)
Vcyl=np.pi*0.2; N=int(r['N']); i5=np.argmin(abs(r['depths']-5.0))
prim_phi=0.5*(np.exp(-0.0653626*5)+np.exp(-0.06122524*5))
k=prim_phi/(r['prim_tl'][i5]/Vcyl/N)
spec_scat_per=r['spec_scat']/Vcyl/N*k
f1=0.5*np.exp(-0.0653626*5); f2=0.5*np.exp(-0.06122524*5)
groups_all=[(E1,f1),(E2,f2)]+[(phc[i],spec_scat_per[i]) for i in range(28) if spec_scat_per[i]>0]
groups_prim=[(E1,f1),(E2,f2)]
groups_scat=[(phc[i],spec_scat_per[i]) for i in range(28) if spec_scat_per[i]>0]

EG=np.exp(np.linspace(np.log(0.010),np.log(1.4),150))
widths=np.gradient(EG)
def build_S0(groups, EG, include_positron=False):
    S0=np.zeros(len(EG))
    for E,w in groups:
        minc=f_incoh(E)
        if minc>0: S0+=w*minc*compton_pdf_grid(E,EG)
        mpe=f_pe(E)
        if mpe>0:
            i=np.argmin(abs(EG-E)); S0[i]+=w*mpe/widths[i]
        mp=f_pair(E)
        if mp>0 and E>1.022:
            Te=(E-1.022)/2; i=np.argmin(abs(EG-Te))
            S0[i]+=(2 if include_positron else 1)*w*mp/widths[i]
    return S0

S0_all=build_S0(groups_all,EG); S0_p=build_S0(groups_prim,EG); S0_s=build_S0(groups_scat,EG)
print("S0 integrals:",np.trapz(S0_all,EG),np.trapz(S0_p,EG),np.trapz(S0_s,EG))
Phi_all,Phi_p0=solve_fluence(S0_all,EG)
Phi_prim,_=solve_fluence(S0_p,EG)
Phi_scat,_=solve_fluence(S0_s,EG)

def fluence_bins(EG,Phi,edges):
    out=np.zeros(len(edges)-1)
    for i in range(len(edges)-1):
        m=(EG>=edges[i])&(EG<edges[i+1])
        out[i]=np.trapz(Phi[m],EG[m]) if m.sum()>1 else 0.0
    return out

L4_spec_all=fluence_bins(EG,Phi_all,EL_EDGES)
L4_flu_tot=L4_spec_all.sum()
L4_spec_p0=fluence_bins(EG,Phi_p0,EL_EDGES)
L4_spec_prim=fluence_bins(EG,Phi_prim,EL_EDGES)
L4_spec_scat=fluence_bins(EG,Phi_scat,EL_EDGES)
print("L4 fluence/Phi0 total:",L4_flu_tot," primary-only(no delta):",L4_spec_p0.sum())
print("delta fraction:",1-L4_spec_p0.sum()/L4_flu_tot)
print("mean electron energy:",np.trapz(EG*Phi_all,EG)/np.trapz(Phi_all,EG))
print("primary-parent fluence:",L4_spec_prim.sum()," scattered-parent:",L4_spec_scat.sum())
Sc=f_Sc(np.maximum(EG,0.011)); Sr=f_Sr(np.maximum(EG,0.011))
print("rad/coll:",np.trapz(Phi_all*Sr,EG)/np.trapz(Phi_all*Sc,EG))
print("L4 spectrum:",np.round(L4_spec_all,5))
np.savez('fluence_results.npz',EG=EG,Phi_all=Phi_all,Phi_prim=Phi_prim,Phi_scat=Phi_scat,Phi_p0=Phi_p0,
         L4_spec_all=L4_spec_all,L4_flu_tot=L4_flu_tot)