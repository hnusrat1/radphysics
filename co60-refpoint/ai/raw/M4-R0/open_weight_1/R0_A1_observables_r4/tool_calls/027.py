import numpy as np
me=0.51099895; re=2.8179403262e-13; NA=6.02214076e23; n_e_g=NA*10/18.01528
def kn_total(E):
    eta=np.asarray(E,float)/me; a=1+2*eta
    return 2*np.pi*re**2*((1+eta)/eta**2*(2*(1+eta)/a-np.log(a)/eta)+np.log(a)/(2*eta)-(1+3*eta)/a**2)
def kn_dsdmu(E,mu):
    eta=E/me; r=1.0/(1.0+eta*(1.0-mu)); return np.pi*re**2*r**2*(r+1.0/r-(1.0-mu*mu))
Egrid=np.array([0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0,3.0])
muR=np.array([0.028,0.012,0.0085,0.006,0.0052,0.0045,0.0033,0.0028,0.0022,0.0019,0.0017,0.0015,0.0012,0.001,0.0009,0.00085,0.0008,0.00075,0.00055,0.0004])
tau=np.array([0.6,0.15,0.045,0.0093,0.0026,0.0019,0.0007,0.00042,0.00016,0.000085,0.000052,0.000036,0.000021,0.000013,0.00001,0.000009,0.000008,0.000007,0.000004,0.0000015])
kap=np.zeros_like(Egrid); sel=Egrid>1.022; kap[sel]=np.interp(np.clip(Egrid[sel],1.1732,3.0),[1.1732,1.3325,1.5,2.0,3.0],[0.00012,0.00075,0.0011,0.0026,0.0055])
def interp1(x,xp,fp): return np.interp(np.clip(x,xp[0],xp[-1]),xp,fp)
def mu_photo(E): return interp1(E,Egrid,tau)
def mu_pair(E):
    E=np.asarray(E,float); return np.where(E>2*me, interp1(E,Egrid,kap),0.0)

d=np.load('l1_data.npz')
sp_p=d['sp_p']; sp_s=d['sp_s']
ph=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
Ec=0.5*(ph[1:]+ph[:-1]); dE=ph[1:]-ph[:-1]
sp_tot=sp_p+sp_s

Se=np.array([0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4])
Scoll=np.array([19.16,10.15,7.19,4.85,4.30,4.13,3.44,2.96,2.43,2.21,2.06,1.97,1.88,1.852,1.833,1.824])
Srad=np.array([0.0002,0.001,0.002,0.004,0.005,0.006,0.008,0.010,0.012,0.015,0.018,0.021,0.027,0.033,0.040,0.047])
def S_col(T): return interp1(T,Se,Scoll)
def S_rad(T): return interp1(T,Se,Srad)

Tg=np.linspace(0.011,1.399,7000); dT=Tg[1]-Tg[0]
S=np.zeros(len(Tg))
for i,E in enumerate(Ec):
    if sp_tot[i]<=0: continue
    eta=E/me; tmax=2*eta/(1+2*eta)
    t=Tg/E; ok=(t<tmax)&(t>0)
    mu=1.0-t[ok]/(eta*(1-t[ok]))
    dsdT=kn_dsdmu(E,mu)/(eta*E*(1-t[ok])**2)
    S[ok]+=sp_tot[i]*n_e_g*dsdT
for i,E in enumerate(Ec):
    if sp_tot[i]<=0: continue
    w=sp_tot[i]*mu_photo(E)
    j=int(np.argmin(np.abs(Tg-E))); S[j]+=w
for i,E in enumerate(Ec):
    if sp_tot[i]<=0: continue
    w=sp_tot[i]*mu_pair(E)
    if w>0:
        Emax=E-1.022
        if Emax>0.01:
            ok=(Tg<=Emax); S[ok]+=w*2.0/Emax

def fluence_from_source(S, Scolv):
    rev = np.cumsum(S[::-1])[::-1]*dT
    return rev/Scolv

Scolv=S_col(Tg)
Phi0 = fluence_from_source(S, Scolv)

def moller_dsdW_vec(Tarr, W):
    out=np.zeros_like(Tarr)
    ok=(Tarr>2*W)&(W<=Tarr/2.0)
    T=Tarr[ok]
    if len(T)==0: return out
    gamma=1.0+T/me; beta2=1.0-1.0/(gamma*gamma)
    out[ok]=2*np.pi*re**2*me/beta2*(1.0/W**2 -1.0/(W*T)+1.0/(T-W)**2 -1.0/((T-W)*T))
    return out

Phi=Phi0.copy()
for it in range(8):
    Sd=np.zeros(len(Tg))
    for j in range(len(Tg)):
        W=Tg[j]
        idx=Tg>W
        if idx.sum()==0: continue
        Sd[j]=np.sum(Phi[idx]*dT*n_e_g*moller_dsdW_vec(Tg[idx],W))
    Phi=fluence_from_source(S+Sd, Scolv)

Phi_tot=Phi
Kcol_check=np.trapezoid(Phi_tot*Scolv, Tg)
print('Kcol from electron fluence (MeV cm2/g):', round(Kcol_check,4), ' expected ~0.0333')
flu_tot=np.trapezoid(Phi_tot,Tg)
flu_no=np.trapezoid(Phi0,Tg)
print('electron fluence/Phi0 no-delta:', round(flu_no,5))
print('electron fluence/Phi0 with delta:', round(flu_tot,5))
print('delta fraction:', round((flu_tot-flu_no)/flu_tot,4))
meanE=np.trapezoid(Phi_tot*Tg,Tg)/flu_tot
print('mean electron energy (MeV):', round(meanE,4))
rat=np.trapezoid(Phi_tot*S_rad(Tg),Tg)/np.trapezoid(Phi_tot*Scolv,Tg)
print('radiative/collision:', round(rat,5))
ebins=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
hist=np.histogram(Tg,bins=ebins,weights=Phi_tot*dT)[0]
print('electron fluence spectrum (per Phi0 per bin):')
print(np.round(hist,6))
print('check sums to flu_tot:', hist.sum())
np.savez('l4_data.npz', flu_tot=flu_tot, delta_frac=(flu_tot-flu_no)/flu_tot, meanE=meanE, rat=rat, spec=hist, Phi_tot=Phi_tot, Tg=Tg)
