import numpy as np
me=0.51099895; re=2.8179403262e-13; NA=6.02214076e23; n_e_g=NA*10/18.01528
def kn_total(E):
    eta=np.asarray(E,float)/me; a=1+2*eta
    return 2*np.pi*re**2*((1+eta)/eta**2*(2*(1+eta)/a-np.log(a)/eta)+np.log(a)/(2*eta)-(1+3*eta)/a**2)
Egrid=np.array([0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0,3.0])
muR=np.array([0.028,0.012,0.0085,0.006,0.0052,0.0045,0.0033,0.0028,0.0022,0.0019,0.0017,0.0015,0.0012,0.001,0.0009,0.00085,0.0008,0.00075,0.00055,0.0004])
tau=np.array([0.6,0.15,0.045,0.0093,0.0026,0.0019,0.0007,0.00042,0.00016,0.000085,0.000052,0.000036,0.000021,0.000013,0.00001,0.000009,0.000008,0.000007,0.000004,0.0000015])
kap=np.zeros_like(Egrid); sel=Egrid>1.022; kap[sel]=np.interp(np.clip(Egrid[sel],1.1732,3.0),[1.1732,1.3325,1.5,2.0,3.0],[0.00012,0.00075,0.0011,0.0026,0.0055])
def interp1(x,xp,fp): return np.interp(np.clip(x,xp[0],xp[-1]),xp,fp)
def mu_comp(E): return kn_total(np.clip(E,0.01,3.0))*n_e_g
def mu_photo(E): return interp1(E,Egrid,tau)
def mu_pair(E):
    E=np.asarray(E,float); return np.where(E>2*me, interp1(E,Egrid,kap),0.0)
d=np.load('l1_data.npz')
sp_p=d['sp_p']; sp_s=d['sp_s']; ang_s=d['ang_s']
ph=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
Ec=0.5*(ph[1:]+ph[:-1])
ct_bins=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
nb=25

# KN quantile
eta_grid=np.linspace(0.02,3.0,400); xi_grid=np.linspace(0.001,0.999,512); mu_grid=np.linspace(-1,1,4001)
def kn_dsdmu_vec(eta,mu):
    r=1.0/(1.0+eta*(1.0-mu)); return np.pi*re**2*r**2*(r+1.0/r-(1.0-mu*mu))
Q=np.zeros((len(eta_grid),len(xi_grid)))
for i,eta in enumerate(eta_grid):
    dd=kn_dsdmu_vec(eta,mu_grid); c=np.cumsum(dd); c=c/c[-1]; Q[i,:]=np.interp(xi_grid,c,mu_grid)
def sample_kn_cos(E,xi):
    eta=np.clip(E/me,eta_grid[0],eta_grid[-1])
    ie=(eta-eta_grid[0])/(eta_grid[-1]-eta_grid[0])*(len(eta_grid)-1)
    i0=np.clip(np.floor(ie).astype(int),0,len(eta_grid)-2); f=ie-i0
    ix=xi*(len(xi_grid)-1); j0=np.clip(np.floor(ix).astype(int),0,len(xi_grid)-2); fx=ix-j0
    v0=Q[i0,j0]*(1-fx)+Q[i0,j0+1]*fx; v1=Q[i0+1,j0]*(1-fx)+Q[i0+1,j0+1]*fx
    return v0*(1-f)+v1*f
def rotate(u,cosT,phi):
    ux,uy,uz=u[...,0],u[...,1],u[...,2]
    e1x=uy; e1y=-ux; e1z=np.zeros_like(ux); n=np.sqrt(e1x*e1x+e1y*e1y)
    e1x=np.where(n>1e-30,e1x/n,1.0); e1y=np.where(n>1e-30,e1y/n,0.0)
    e2x=uy*e1z-uz*e1y; e2y=uz*e1x-ux*e1z; e2z=ux*e1y-uy*e1x
    st=np.sqrt(np.clip(1-cosT*cosT,0,1)); cphi=np.cos(phi); sphi=np.sin(phi)
    nx=e1x*(st*cphi)+e2x*(st*sphi)+ux*cosT; ny=e1y*(st*cphi)+e2y*(st*sphi)+uy*cosT; nz=e1z*(st*cphi)+e2z*(st*sphi)+uz*cosT
    return np.stack([nx,ny,nz],axis=-1)

rng=np.random.default_rng(7)
N=3_000_000
# sample photon energy weighted by fluence*mu_comp (electron production rate)
w_E = (sp_p+sp_s)*mu_comp(Ec)
w_E/=w_E.sum()
Eph = rng.choice(Ec, size=N, p=w_E)
# sample primary vs scattered
# for each sampled photon: if it's a primary-line photon -> direction cos=1; else sample from angular dist of scattered (use ang_s)
is_prim_line = (Eph>1.15)&(Eph<1.335)  # rough
# Actually we should decide prim/scat by fluence: use prob prim = sp_p-weighted/(total-weighted)
w_p = sp_p*mu_comp(Ec); w_s=sp_s*mu_comp(Ec)
pp = w_p/(w_p+w_s+1e-30)
pr = rng.random(N)<pp[np.searchsorted(Ec,Eph)]
# direction cos of photon for scattered: sample from ang_s distribution
cosg = np.where(pr, 1.0, rng.choice(ct_bins[:-1]+0.1, size=N, p=ang_s)+0.0)  # sample bin midpoints
# smoother: sample from piecewise uniform in bins
u_s = rng.random(N)
edges = ct_bins
cum=np.concatenate([[0],np.cumsum(ang_s)])
idx=np.searchsorted(cum,u_s)-1
cosg=np.where(pr,1.0,edges[idx]+(u_s-cum[idx])/(cum[idx+1]-cum[idx]+1e-30)*0.2-0.9-0.1)  # approximate continuous within bin
# let me simply use bin center for scattered directions
cosg=np.where(pr,1.0, edges[idx]+0.1)
# photon direction u with that cos and random azimuth
phi_g = rng.random(N)*2*np.pi
sint=np.sqrt(np.clip(1-cosg*cosg,0,1))
u=np.stack([sint*np.cos(phi_g), sint*np.sin(phi_g), cosg],axis=-1)
# sample Compton scattering angle
cosT=sample_kn_cos(Eph, rng.random(N)); phi=2*np.pi*rng.random(N)
up=rotate(u, cosT, phi)
r_ratio=1.0/(1.0+(Eph/me)*(1.0-cosT))  # E'/E
evec = u - r_ratio[:,None]*up
nrm=np.sqrt((evec*evec).sum(axis=1)+1e-300)
ce = evec[:,2]/nrm
# photoelectric and pair: direction = photon direction; add a few percent
w_ph = mu_photo(Eph)/(mu_comp(Eph)+mu_photo(Eph)+mu_pair(Eph))
w_pa = mu_pair(Eph)/(mu_comp(Eph)+mu_photo(Eph)+mu_pair(Eph))
r2=rng.random(N)
ce = np.where(r2< (w_ph+w_pa), cosg, ce)
hist=np.histogram(ce, bins=ct_bins)[0]
hist=hist/hist.sum()
print('L3 initial direction:', np.round(hist,5))
print('sum',hist.sum())
np.save('l3_dir.npy',hist)
