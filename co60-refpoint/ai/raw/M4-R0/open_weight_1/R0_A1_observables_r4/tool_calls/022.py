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
def mu_comp(E): return kn_total(np.clip(E,0.01,3.0))*n_e_g
def mu_ray(E): return interp1(E,Egrid,muR)
def mu_photo(E): return interp1(E,Egrid,tau)
def mu_pair(E):
    E=np.asarray(E,float); return np.where(E>2*me, interp1(E,Egrid,kap),0.0)
def mu_totX(E): return mu_comp(E)+mu_ray(E)+mu_photo(E)+mu_pair(E)

d=np.load('l1_data.npz')
sp_p=d['sp_p']; sp_s=d['sp_s']
ph=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
Ec=0.5*(ph[1:]+ph[:-1])

muq=np.linspace(-1,1,20001)
def comp_etr(E):
    dds=kn_dsdmu(E,muq)
    t=E*(E/me)*(1-muq)/(1+(E/me)*(1-muq))
    return np.trapezoid(dds*t,muq)/np.trapezoid(dds,muq)

def inter_fractions(sp):
    mc=(sp*mu_comp(Ec)).sum(); mr=(sp*mu_ray(Ec)).sum(); mp=(sp*mu_photo(Ec)).sum(); mpa=(sp*mu_pair(Ec)).sum()
    tot=mc+mr+mp+mpa
    return dict(compton=mc/tot, rayleigh=mr/tot, photoelectric=mp/tot, pair=mpa/tot), (mc,mr,mp,mpa,tot)

def etr_fractions(sp):
    ec=sum((sp[i]*mu_comp(Ecb)*comp_etr(Ecb) for i,Ecb in enumerate(Ec)))
    ep=sum((sp[i]*mu_photo(Ecb)*Ecb for i,Ecb in enumerate(Ec)))
    epa=sum((sp[i]*mu_pair(Ecb)*max(Ecb-1.022,0) for i,Ecb in enumerate(Ec)))
    er=0.0
    tot=ec+ep+epa+er
    return dict(compton=ec/tot, rayleigh=er/tot, photoelectric=ep/tot, pair=epa/tot)

f_all,_=inter_fractions(sp_p+sp_s)
f_pri,_=inter_fractions(sp_p)
f_sca,_=inter_fractions(sp_s)
f_etr=etr_fractions(sp_p+sp_s)
print('interaction fractions all:', {k:round(v,5) for k,v in f_all.items()})
print('interaction fractions primary:', {k:round(v,5) for k,v in f_pri.items()})
print('interaction fractions scattered:', {k:round(v,5) for k,v in f_sca.items()})
print('energy transfer fractions all:', {k:round(v,5) for k,v in f_etr.items()})
_, a = inter_fractions(sp_p+sp_s)
_, b = inter_fractions(sp_p)
_, c = inter_fractions(sp_s)
print('scattered share of interactions:', c[4]/a[4])
