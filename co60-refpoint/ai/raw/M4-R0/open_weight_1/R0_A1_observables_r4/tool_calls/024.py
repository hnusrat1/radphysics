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
ebins=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])

def source_spectrum(sp, Tg):
    dT=Tg[1]-Tg[0]; S=np.zeros(len(Tg))
    for i,E in enumerate(Ec):
        if sp[i]<=0: continue
        eta=E/me
        tmax=2*eta/(1+2*eta)
        t=Tg/E
        ok=(Tg>1e-6)&(t<tmax)&(t>0)
        mu=1.0-t[ok]/(eta*(1-t[ok]))
        dsdT=kn_dsdmu(E,mu)/(eta*E*(1-t[ok])**2)
        S[ok]+=sp[i]/dE[i]*n_e_g*dsdT
    for i,E in enumerate(Ec):
        if sp[i]<=0: continue
        w=sp[i]/dE[i]*mu_photo(E)
        j=int(np.argmin(np.abs(Tg-E))); S[j]+=w
    for i,E in enumerate(Ec):
        if sp[i]<=0: continue
        w=sp[i]/dE[i]*mu_pair(E)
        if w>0:
            Emax=max(E-1.022,0)
            if Emax>0:
                ok=(Tg>=0)&(Tg<=Emax); S[ok]+=w*2.0/Emax
    hist,_=np.histogram(Tg,bins=ebins,weights=S*dT)
    return hist

Tg=np.linspace(0.0005,1.3995,14000)
hist_all=source_spectrum(sp_p+sp_s,Tg)
hist_prim=source_spectrum(sp_p,Tg)
Ec_e=0.5*(ebins[1:]+ebins[:-1])
print('initial spectrum ALL:', np.round(hist_all/hist_all.sum(),5))
print('initial spectrum PRIMARY:', np.round(hist_prim/hist_prim.sum(),5))
mean_all=(hist_all*Ec_e).sum()/hist_all.sum()
mean_prim=(hist_prim*Ec_e).sum()/hist_prim.sum()
print('mean initial energy all', round(mean_all,4), 'primary-only', round(mean_prim,4))
# verify mean recoil for a single 1.1732 photon:
from scipy.integrate import quad
E0=1.1732; eta=E0/me; tmax=2*eta/(1+2*eta)
def dsdT(t):
    mu=1-t/(eta*(1-t))
    return kn_dsdmu(E0,mu)/(eta*E0*(1-t)**2)
mT=quad(lambda t: t*E0*dsdT(t),0,tmax,limit=200)[0]/quad(dsdT,0,tmax,limit=200)[0]
print('mean recoil for 1.1732 single:',mT)
np.save('l3_hist.npy',hist_all); np.save('l3_hist_prim.npy',hist_prim)
