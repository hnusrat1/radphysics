
mc = r'''
import numpy as np, json
me = 0.51099895
NA_e = 3.3428e23      # electrons / cm^3 (water)
N_mol = 3.3428e22     # H2O molecules / cm^3
r_e = 2.8179403e-13
rng = np.random.default_rng(2024)
ENERGIES = np.array([1.1732, 1.3325])
ZMAX=30.0; HALF=15.0
Z_SC0, Z_SC1 = 4.9, 5.1
R_SC = 1.0
E_CUT = 0.010   # photons below 10 keV absorbed (photoelectric)

def kn_sigma(E):
    k = E/me
    return 2*np.pi*r_e**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k) + np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2)

_th = np.linspace(1e-9, np.pi, 400001)
def rayleigh_sigma(E):
    lam = 12.398/(E*1000)
    s = np.sin(_th/2)/lam
    F2 = 100/(1+(s/0.3)**2)**4
    f = 0.5*(1+np.cos(_th)**2)*F2*2*np.pi*np.sin(_th)
    return r_e**2*np.trapezoid(f, _th)

def photo_sigma_per_g(E):
    # approximate water photoelectric: 2.8e-3 cm2/g at 100 keV, E^-3 scaling
    return 2.8e-3*(0.1/E)**3

def pair_mu(E):
    # rough near-threshold pair attenuation (cm^-1)
    return np.where(E>1.022, 0.0001*np.clip((E-1.022)/0.2,0,1), 0.0)

# tables on log grid
EG = np.logspace(np.log10(E_CUT), np.log10(1.4), 3000)
MI_G = NA_e*kn_sigma(EG)
MC_G = np.array([N_mol*rayleigh_sigma(e) for e in EG])
MP_G = np.array([N_mol*0 + 1.0*photo_sigma_per_g(e)  for e in EG])  # cm2/g * 1 g/cm3
MPR_G = pair_mu(EG)
def mu_tabs(E):
    lE = np.log(np.clip(E,E_CUT,1.4))
    lg = np.log(EG)
    mi = np.interp(lE, lg, MI_G); mc = np.interp(lE, lg, MC_G)
    mp = np.interp(lE, lg, MP_G); mpr = np.interp(lE, lg, MPR_G)
    return mi, mc, mp, mpr

def rot_dir(u, v, w, cth, phi):
    sth = np.sqrt(np.maximum(0.0, 1-cth**2))
    use_x = np.abs(w) >= 0.9
    rx = np.where(use_x,0.0,1.0); ry = np.where(use_x,1.0,0.0); rz=np.zeros_like(u)
    # e1 = normalize(d x r)
    e1x = v*rz - w*ry; e1y = w*rx - u*rz; e1z = u*ry - v*rx
    n1 = np.sqrt(e1x**2+e1y**2+e1z**2); e1x/=n1; e1y/=n1; e1z/=n1
    e2x = v*e1z - w*e1y; e2y = w*e1x - u*e1z; e2z = u*e1y - v*e1x
    cp=np.cos(phi); sp=np.sin(phi)
    nx = sth*(cp*e1x+sp*e2x)+cth*u
    ny = sth*(cp*e1y+sp*e2y)+cth*v
    nz = sth*(cp*e1z+sp*e2z)+cth*w
    nn=np.sqrt(nx**2+ny**2+nz**2)
    return nx/nn, ny/nn, nz/nn

def sample_KN_vec(E):
    out=np.empty(E.size); k=E/me; todo=np.arange(E.size)
    while todo.size:
        kk=k[todo]; m=todo.size
        c=2*rng.random(m)-1
        eps=1/(1+kk*(1-c))
        f=eps**2*(eps+1/eps-(1-c**2))
        acc=rng.random(m)*2<f
        out[todo[acc]]=c[acc]; todo=todo[~acc]
    return out

def sample_rayl_vec(E):
    lam = 12.398/(E*1000)
    U = (1-rng.random(E.size))**(-1/3)-1
    s = 0.3*np.sqrt(U)
    sh = np.clip(lam*s,0,1)
    return 1-2*sh**2

def sample_pe_vec(E):
    b_all = np.sqrt(1-1/(1+E/me)**2)
    out=np.empty(E.size); todo=np.arange(E.size)
    while todo.size:
        b=b_all[todo]; m=todo.size
        c=2*rng.random(m)-1
        f=(1-c**2)/(1-b*c)**4
        fmax=1/(1-b)**4
        acc=rng.random(m)*fmax<f
        out[todo[acc]]=c[acc]; todo=todo[~acc]
    return out

def isotropic(n):
    c = 2*rng.random(n)-1; phi = 2*np.pi*rng.random(n)
    s = np.sqrt(1-c**2)
    return s*np.cos(phi), s*np.sin(phi), c

def in_volume(x,y,z):
    return (x*x+y*y <= R_SC**2) & (z>=Z_SC0) & (z<=Z_SC1)

def run(n_per_E):
    events=[]
    for E0 in ENERGIES:
        n=n_per_E
        xs=[];ys=[];got=0
        while got<n:
            m=int((n-got)*1.3)+100
            x=(2*rng.random(m)-1)*0.05; y=(2*rng.random(m)-1)*0.05
            keep=rng.random(m)<(1+x*x+y*y)**(-1.5)
            xs.append(x[keep]); ys.append(y[keep]); got+=keep.sum()
        x=np.concatenate(xs)[:n]; y=np.concatenate(ys)[:n]
        d=np.sqrt(1+x*x+y*y)
        u=x/d; v=y/d; w=1/d
        t=(0-(-100.0))/w
        P=dict(x=u*t, y=v*t, z=np.zeros(n), u=u, v=v, w=w, E=np.full(n,E0))
        stack=[P]
        while stack:
            P=stack.pop()
            x=P['x'];y=P['y'];z=P['z'];u=P['u'];v=P['v'];w=P['w'];E=P['E']
            if x.size==0: continue
            mi,mc,mp,mpr = mu_tabs(E)
            mtot = mi+mc+mp+mpr
            s = -np.log(1-rng.random(x.size))/mtot
            xn=x+u*s; yn=y+v*s; zn=z+w*s
            inside=(np.abs(xn)<=HALF)&(np.abs(yn)<=HALF)&(zn>=0)&(zn<=ZMAX)
            idx=np.where(inside)[0]
            if idx.size==0: continue
            x=xn[idx]; y=yn[idx]; z=zn[idx]; u=u[idx]; v=v[idx]; w=w[idx]; Ei=E[idx]
            mi,mc,mp,mpr=mi[idx],mc[idx],mp[idx],mpr[idx]
            m=idx.size
            r=rng.random(m)*(mi+mc+mp+mpr)
            proc=np.where(r<mi,0,np.where(r<mi+mc,1,np.where(r<mi+mc+mp,3,2)))  # 0 compton,1 rayleigh,2 pair,3 photo
            insc=in_volume(x,y,z)
            # Compton
            ic=np.where(proc==0)[0]
            if ic.size:
                Eic=Ei[ic]
                cth=sample_KN_vec(Eic)
                Eout=Eic/(1+Eic/me*(1-cth))
                phi=2*np.pi*rng.random(ic.size)
                nu,nv,nw=rot_dir(u[ic],v[ic],w[ic],cth,phi)
                te=Eic-Eout
                pm=np.sqrt(np.maximum(Eic**2+Eout**2-2*Eic*Eout*cth,1e-30))
                cos_e=np.clip((Eic-Eout*cth)/pm,-1,1)
                for j in np.where(insc[ic])[0]:
                    events.append(dict(process='compton', e_in_MeV=float(Eic[j]), e_out_MeV=float(Eout[j]),
                        cos_scatter=float(cth[j]), t_e_MeV=float(te[j]), cos_electron=float(cos_e[j]), t_positron_MeV=None))
                stack.append(dict(x=x[ic],y=y[ic],z=z[ic],u=nu,v=nv,w=nw,E=Eout))
            # Rayleigh
            ir=np.where(proc==1)[0]
            if ir.size:
                cth=sample_rayl_vec(Ei[ir]); phi=2*np.pi*rng.random(ir.size)
                nu,nv,nw=rot_dir(u[ir],v[ir],w[ir],cth,phi)
                for j in np.where(insc[ir])[0]:
                    events.append(dict(process='rayleigh', e_in_MeV=float(Ei[ir][j]), e_out_MeV=float(Ei[ir][j]),
                        cos_scatter=float(cth[j]), t_e_MeV=0.0, cos_electron=None, t_positron_MeV=None))
                stack.append(dict(x=x[ir],y=y[ir],z=z[ir],u=nu,v=nv,w=nw,E=Ei[ir]))
            # Pair
            ip=np.where(proc==2)[0]
            if ip.size:
                Ep=Ei[ip]; Tt=Ep-2*me
                frac=rng.random(ip.size)
                te_k=Tt*frac; tp_k=Tt*(1-frac)
                ang=np.abs(rng.normal(0,1,ip.size))*me/(te_k+me)
                cos_el=np.cos(np.minimum(ang,np.pi))
                for j in np.where(insc[ip])[0]:
                    events.append(dict(process='pair', e_in_MeV=float(Ep[j]), e_out_MeV=None, cos_scatter=None,
                        t_e_MeV=float(te_k[j]), cos_electron=float(cos_el[j]), t_positron_MeV=float(tp_k[j])))
                ax,ay,az=isotropic(ip.size)
                stack.append(dict(x=np.concatenate([x[ip],x[ip]]), y=np.concatenate([y[ip],y[ip]]), z=np.concatenate([z[ip],z[ip]]),
                    u=np.concatenate([ax,-ax]), v=np.concatenate([ay,-ay]), w=np.concatenate([az,-az]),
                    E=np.full(2*ip.size, me)))
            # Photoelectric
            ipe=np.where(proc==3)[0]
            if ipe.size:
                Epe=Ei[ipe]; cpe=sample_pe_vec(Epe)
                for j in np.where(insc[ipe])[0]:
                    events.append(dict(process='photoelectric', e_in_MeV=float(Epe[j]), e_out_MeV=None, cos_scatter=None,
                        t_e_MeV=float(Epe[j]), cos_electron=float(cpe[j]), t_positron_MeV=None))
    return events
'''
open('mc.py','w').write(mc)
test = r'''
import time
exec(open("mc.py").read())
t=time.time()
ev = run(200000)
print("time", time.time()-t, "n events", len(ev))
from collections import Counter
print(Counter(e['process'] for e in ev))
print(Counter((e['e_in_MeV']>0.6, e['process']) for e in ev))
print(mu_tabs(np.array([1.1732,1.3325,0.5,0.2])))
'''
open('test.py','w').write(test)
import subprocess
r=subprocess.run(['python','test.py'],capture_output=True,text=True); print(r.stdout, r.stderr[-3000:])
