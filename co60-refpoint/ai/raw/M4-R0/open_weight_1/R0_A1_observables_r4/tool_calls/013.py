
# Debug: instrument the photon loop for a small run to find why primary track is too large
import numpy as np, time
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
def mu_ray(E): return interp1(E,Egrid,muR)
def mu_photo(E): return interp1(E,Egrid,tau)
def mu_pair(E):
    E=np.asarray(E,float); return np.where(E>2*me, interp1(E,Egrid,kap),0.0)
def mu_totX(E): return mu_comp(E)+mu_ray(E)+mu_photo(E)+mu_pair(E)

def score_cylinder(x,y,z,ux,uy,uz,s):
    zlo=4.9; zhi=5.1; rad=1.0
    tz1=(zlo-z)/uz; tz2=(zhi-z)/uz
    tza=np.minimum(tz1,tz2); tzb=np.maximum(tz1,tz2)
    tza=np.clip(tza,0.0,s); tzb=np.clip(tzb,0.0,s)
    uz0=np.abs(uz)<1e-9
    tza=np.where(uz0,0.0,tza); tzb=np.where(uz0,s,tzb)
    tza=np.where(uz0&((z<zlo)|(z>zhi)),0.0,tza); tzb=np.where(uz0&((z<zlo)|(z>zhi)),0.0,tzb)
    a=ux*ux+uy*uy; b=2*(x*ux+y*uy); c=x*x+y*y-rad*rad
    disc=b*b-4*a*c; a0=np.abs(a)<1e-9
    sq=np.sqrt(np.clip(disc,0,None)); den=2*a+1e-30
    t1=(-b-sq)/den; t2=(-b+sq)/den; neg=disc<0
    tlo=np.where(neg,0.0,np.minimum(t1,t2)); thi=np.where(neg,np.where(c<0,s,0.0),np.maximum(t1,t2))
    start_in=c<0
    tlo=np.where(start_in&~neg&~a0,0.0,tlo); thi=np.where(start_in&~neg&~a0,np.maximum(t1,t2),thi)
    tlo=np.where(a0,0.0,tlo); thi=np.where(a0,np.where(c<0,s,0.0),thi)
    tlo=np.maximum(tlo,tza); thi=np.minimum(thi,tzb)
    return np.maximum(thi-tlo,0.0)

rng=np.random.default_rng(3)
N=300000
x0=rng.uniform(-5,5,N); y0=rng.uniform(-5,5,N)
d=np.sqrt(x0*x0+y0*y0+1e4)
ux=x0/d; uy=y0/d; uz=100.0/d
E=np.where(rng.random(N)<0.5,1.1732,1.3325)
x=np.zeros(N); y=np.zeros(N); z=np.zeros(N); scat=np.zeros(N,bool); alive=np.ones(N,bool)
Hcnt=np.zeros(2); nscore_prim=0; nscore_scat=0
for it in range(20):
    Ncur=len(alive)
    mu=mu_totX(E)
    s=-np.log(rng.random(Ncur))/mu
    s=np.where(alive,s,0.0)
    tex=np.full(Ncur,np.inf)
    tz=(30-z)/uz; ok=uz>0; tex=np.where(ok,np.minimum(tex,np.where(tz>0,tz,np.inf)),tex)
    tz=(0-z)/uz; ok=uz<0; tex=np.where(ok,np.minimum(tex,np.where(tz>0,tz,np.inf)),tex)
    tx=(15-x)/ux; ok=ux>0; tex=np.where(ok,np.minimum(tex,np.where(tx>0,tx,np.inf)),tex)
    tx=(-15-x)/ux; ok=ux<0; tex=np.where(ok,np.minimum(tex,np.where(tx>0,tx,np.inf)),tex)
    ty=(15-y)/uy; ok=uy>0; tex=np.where(ok,np.minimum(tex,np.where(ty>0,ty,np.inf)),tex)
    ty=(-15-y)/uy; ok=uy<0; tex=np.where(ok,np.minimum(tex,np.where(ty>0,ty,np.inf)),tex)
    s_eff=np.minimum(s,tex); s_eff=np.where(alive,s_eff,0.0)
    tl=score_cylinder(x,y,z,ux,uy,uz,s_eff)*alive
    Hcnt[0]+=tl[~scat].sum(); Hcnt[1]+=tl[scat].sum()
    nscore_prim+=(tl>0)[~scat].sum(); nscore_scat+=(tl>0)[scat].sum()
    interact=(s<tex)&alive
    dead=alive&(s>=tex)
    x=x+ux*s*alive; y=y+uy*s*alive; z=z+uz*s*alive
    if interact.any():
        ind=np.where(interact)[0]
        # simple: treat all as Compton scatter with fixed mu=-0.9? no: mark scattered
        # just mark scat and keep direction (cheat for debug)
        scat[ind]=True
        # kill photons that would go below threshold... keep all alive
        dead[ind]=False
    alive=(~dead)&alive
    if not alive.any(): break
print("Hcnt",Hcnt,"nscore_prim",nscore_prim,"nscore_scat",nscore_scat,"alive left",alive.sum())
print("mean primary track per photon",Hcnt[0]/N)
# expectation: photons reaching the cylinder unscattered with radius <=1
frac_rad = np.pi*((1/1.05)**2)/100.0
exp_n = N*np.exp(-mu_totX(1.25)*5)*frac_rad
print("expected crossing photons (atten ~1.25):",exp_n, "track exp",exp_n*0.2)
