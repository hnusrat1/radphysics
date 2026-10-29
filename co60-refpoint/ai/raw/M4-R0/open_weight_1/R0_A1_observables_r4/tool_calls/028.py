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
Ec=0.5*(ph[1:]+ph[:-1])
ebins=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
nb=len(ebins)-1
Emid=0.5*(ebins[1:]+ebins[:-1])

def initial_bins(sp):
    W=np.zeros(nb)
    # Compton: integrate dsigma/dmu over mu range of each electron bin
    mug=np.linspace(-1,1,20001)
    for i,E in enumerate(Ec):
        if sp[i]<=0: continue
        eta=E/me
        # T = E*eta*(1-mu)/(1+eta*(1-mu)) ; as function of mu
        T = E*eta*(1-mug)/(1+eta*(1-mug))
        dsdmu = kn_dsdmu(E,mug)
        # distribution of T: weight dsdmu
        # histogram T into ebins with weights dsdmu*dmu
        h,_=np.histogram(T, bins=ebins, weights=dsdmu*(mug[1]-mug[0]))
        W += sp[i]*n_e_g*h
    # photoelectric: lump at E
    for i,E in enumerate(Ec):
        if sp[i]<=0: continue
        j=np.clip(np.searchsorted(ebins,E,side='right')-1,0,nb-1)
        W[j]+= sp[i]*mu_photo(E)
    # pair: uniform two particles over [0,E-1.022]
    for i,E in enumerate(Ec):
        if sp[i]<=0: continue
        w2=sp[i]*mu_pair(E)
        if w2<=0: continue
        Emax=E-1.022
        if Emax<=0.01: continue
        for j in range(nb):
            lo=max(ebins[j],0.0); hi=min(ebins[j+1],Emax)
            if hi>lo:
                W[j]+= w2*2.0*(hi-lo)/Emax
    return W
W_all=initial_bins(sp_p+sp_s)
W_prim=initial_bins(sp_p)
print('L3 initial spectrum (all):', np.round(W_all/W_all.sum(),5))
print('L3 initial spectrum (primary):', np.round(W_prim/W_prim.sum(),5))
print('mean initial all:', round((W_all*Emid).sum()/W_all.sum(),4))
print('mean initial primary:', round((W_prim*Emid).sum()/W_prim.sum(),4))
np.save('l3_bins_all.npy',W_all); np.save('l3_bins_prim.npy',W_prim)
