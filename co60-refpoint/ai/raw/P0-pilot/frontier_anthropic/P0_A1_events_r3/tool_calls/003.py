
code = r'''
import numpy as np, sys
rng=np.random.default_rng(12345)
re=2.8179403e-13; me=0.51099895
NA=6.02214076e23
ne=(2*1+8)/18.015*NA
def sigKN(E):
    k=E/me; L=np.log(1+2*k)
    return 2*np.pi*re**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-L/k)+L/(2*k)-(1+3*k)/(1+2*k)**2)
def mu_comp(E): return ne*sigKN(E)
def mu_pair(E): return np.where(E>2*me, 0.00048*np.minimum(1,np.maximum(E-2*me,0)/(2-2*me)),0.0)
def mu_coh(E): return 0.0051*(0.1/E)**2
def mu_pe(E): return 2.76e-3*(0.1/E)**3

def sample_KN(E,n):
    k=E/me
    out=np.empty(n); todo=np.arange(n)
    mg=np.linspace(-1,1,20001); e=1/(1+k*(1-mg)); fmax=(e**2*(e+1/e-(1-mg**2))).max()*1.02
    while todo.size:
        m=rng.uniform(-1,1,todo.size); e=1/(1+k*(1-m))
        f=e**2*(e+1/e-(1-m**2))
        acc=rng.uniform(0,fmax,todo.size)<f
        out[todo[acc]]=m[acc]; todo=todo[~acc]
    return out
def sample_rayleigh_mu(n):
    out=np.empty(n); todo=np.arange(n)
    while todo.size:
        m=rng.uniform(-1,1,todo.size)
        acc=rng.uniform(0,1,todo.size)<(1+m**2)/2
        out[todo[acc]]=m[acc]; todo=todo[~acc]
    return out
def rotate(d, mu, phi):
    n=d.shape[0]
    a=np.where(np.abs(d[:,0:1])<0.9, np.array([[1.,0,0]]), np.array([[0,1.,0]]))
    e1=np.cross(d,a); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(d,e1)
    s=np.sqrt(np.clip(1-mu**2,0,None))
    return d*mu[:,None]+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
def iso_dirs(n):
    c=rng.uniform(-1,1,n); p=rng.uniform(0,2*np.pi,n); s=np.sqrt(1-c*c)
    return np.stack([s*np.cos(p),s*np.sin(p),c],1)
def start_photons(N):
    xs=[]; cnt=0; L=100.0
    while cnt<N:
        m=2*N
        x=rng.uniform(-5,5,m); y=rng.uniform(-5,5,m)
        r=np.sqrt(L*L+x*x+y*y)
        acc=rng.uniform(0,1,m)<(L/r)**3
        xs.append(np.stack([x[acc],y[acc],np.full(acc.sum(),L)],1)); cnt+=acc.sum()
    v=np.concatenate(xs)[:N]
    d=v/np.linalg.norm(v,axis=1)[:,None]
    t=100.0/d[:,2]
    p=np.zeros((N,3))+np.array([0,0,-100.0])+t[:,None]*d
    p[:,2]=0.0
    return p,d

LO=np.array([-15.,-15.,0.]); HI=np.array([15.,15.,30.])
E0=2.0
NHIST=int(sys.argv[1]) if len(sys.argv)>1 else 1000000

# event storage
ev=[]  # tuples: process, e_in, e_out, cos_s, t_e, cos_e, t_pos
def record(proc, ein, eout, cs, te, ce, tp, mask):
    idx=np.nonzero(mask)[0]
    for i in idx:
        ev.append((proc, float(ein[i]), None if eout is None else float(eout[i]),
                   None if cs is None else float(cs[i]), float(te[i]),
                   None if ce is None else float(ce[i]), None if tp is None else float(tp[i])))

def inside_volume(p):
    r2=p[:,0]**2+p[:,1]**2
    return (r2<=1.0) & (p[:,2]>=9.9) & (p[:,2]<=10.1)

# Initial photons
p,d=start_photons(NHIST)
E=np.full(NHIST,E0)
prim=np.ones(NHIST,bool)
# particle queue: pos, dir, E
pos=p; dr=d; En=E.copy()
stats={'compton':0,'rayleigh':0,'pair':0,'photo':0,'escape':0,'annih':0}
it=0
while pos.shape[0]>0:
    it+=1
    n=pos.shape[0]
    # distance to exit
    tex=np.full(n,np.inf)
    for ax in range(3):
        u=dr[:,ax]
        with np.errstate(divide='ignore',invalid='ignore'):
            tp_=np.where(u>0,(HI[ax]-pos[:,ax])/u,np.where(u<0,(LO[ax]-pos[:,ax])/u,np.inf))
        tex=np.minimum(tex,tp_)
    tex=np.maximum(tex,0)
    mc=mu_comp(En); mp=mu_pair(En); mr=mu_coh(En); mpe=mu_pe(En)
    mt=mc+mp+mr+mpe
    s=-np.log(rng.uniform(0,1,n))/mt
    inter=s<tex
    # escaping photons: stats only
    stats['escape']+=int((~inter).sum())
    pos=pos[inter]; dr=dr[inter]; En=En[inter]; mc=mc[inter]; mp=mp[inter]; mr=mr[inter]; mpe=mpe[inter]; mt=mt[inter]; s=s[inter]
    if pos.shape[0]==0: break
    pos=pos+s[:,None]*dr
    m=pos.shape[0]
    inv=inside_volume(pos)
    xi=rng.uniform(0,1,m)*mt
    isC=xi<mc; isP=(~isC)&(xi<mc+mp); isR=(~isC)&(~isP)&(xi<mc+mp+mr)
    isPE=~(isC|isP|isR)
    newpos=[]; newdir=[]; newE=[]
    # Compton
    idx=np.nonzero(isC)[0]
    if idx.size:
        Ei=En[idx]; mu=sample_KN(1.0,0) if False else None
        mu=np.empty(idx.size)
        for E_val in np.unique(Ei):
            sel=Ei==E_val
            mu[sel]=sample_KN(E_val,sel.sum())
        Eo=Ei/(1+(Ei/me)*(1-mu))
        phi=rng.uniform(0,2*np.pi,idx.size)
        dd=dr[idx]
        ndir=rotate(dd,mu,phi)
        pe_par=Ei-Eo*mu
        pe_mag=np.sqrt(Ei**2+Eo**2-2*Ei*Eo*mu)
        ce=pe_par/pe_mag
        te=Ei-Eo
        mask=inv[idx]
        # record
        for j in np.nonzero(mask)[0]:
            ev.append(('compton',float(Ei[j]),float(Eo[j]),float(mu[j]),float(te[j]),float(ce[j]),None))
        stats['compton']+=idx.size
        newpos.append(pos[idx]); newdir.append(ndir); newE.append(Eo)
    # Rayleigh
    idx=np.nonzero(isR)[0]
    if idx.size:
        mu=sample_rayleigh_mu(idx.size); phi=rng.uniform(0,2*np.pi,idx.size)
        ndir=rotate(dr[idx],mu,phi)
        mask=inv[idx]
        for j in np.nonzero(mask)[0]:
            ev.append(('rayleigh',float(En[idx][j]),float(En[idx][j]),float(mu[j]),0.0,None,None))
        stats['rayleigh']+=idx.size
        newpos.append(pos[idx]); newdir.append(ndir); newE.append(En[idx])
    # pair
    idx=np.nonzero(isP)[0]
    if idx.size:
        Ei=En[idx]; T=Ei-2*me
        fr=rng.uniform(0,1,idx.size)
        Te=T*fr; Tp=T*(1-fr)
        # angles: approximate, scale m/(E_kin+m) times |N(0,1)|
        th_e=np.abs(rng.normal(0,1,idx.size))*me/(Te+me)
        th_p=np.abs(rng.normal(0,1,idx.size))*me/(Tp+me)
        ce=np.cos(np.minimum(th_e,np.pi))
        mask=inv[idx]
        for j in np.nonzero(mask)[0]:
            ev.append(('pair',float(Ei[j]),None,None,float(Te[j]),float(ce[j]),float(Tp[j])))
        stats['pair']+=idx.size
        # annihilation photons (at rest), two 0.511 MeV back to back
        dirs=iso_dirs(idx.size)
        newpos.append(pos[idx]); newdir.append(dirs); newE.append(np.full(idx.size,me))
        newpos.append(pos[idx]); newdir.append(-dirs); newE.append(np.full(idx.size,me))
        stats['annih']+=2*idx.size
    # photoelectric
    idx=np.nonzero(isPE)[0]
    if idx.size:
        Ei=En[idx]; mask=inv[idx]
        th=np.abs(rng.normal(0,1,idx.size))*me/(Ei+me)
        ce=np.cos(np.minimum(th,np.pi))
        for j in np.nonzero(mask)[0]:
            ev.append(('photoelectric',float(Ei[j]),None,None,float(Ei[j]),float(ce[j]),None))
        stats['photo']+=idx.size
    if newpos:
        pos=np.concatenate(newpos); dr=np.concatenate(newdir); En=np.concatenate(newE)
        # annihilation photons born at rest; for others keep
    else:
        pos=np.zeros((0,3)); dr=np.zeros((0,3)); En=np.zeros(0)
    # for Rayleigh/Compton the photon's new direction already set; pair annihilation photons have energy me already
print("iterations",it,"stats",stats,"nevents",len(ev), file=sys.stderr)
import pickle
pickle.dump(ev,open('events_%d.pkl'%NHIST,'wb'))
'''
open('mc.py','w').write(code)
import subprocess,time
t=time.time()
r=subprocess.run(['python','mc.py','200000'],capture_output=True,text=True)
print(r.stdout, r.stderr[-3000:], time.time()-t)
