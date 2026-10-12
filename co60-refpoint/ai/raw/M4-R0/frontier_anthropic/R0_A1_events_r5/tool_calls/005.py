
mc = r'''
import numpy as np, json, sys
me = 0.51099895
NA_e = 3.3428e23
N_mol = 3.3428e22
r_e = 2.8179403e-13
MU_PAIR_CM = 0.0001
MU_PE_CM = 0.0
rng = np.random.default_rng(2024)
ENERGIES = [1.1732, 1.3325]

def kn_sigma(E):
    k = E/me
    return 2*np.pi*r_e**2*((1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k) + np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2)

def rayleigh_sigma(lam_A):
    a = 0.3
    th = np.linspace(1e-9, np.pi, 200001)
    s = np.sin(th/2)/lam_A
    F2 = 100/(1+(s/a)**2)**4
    f = 0.5*(1+np.cos(th)**2)*F2*2*np.pi*np.sin(th)
    return r_e**2*np.trapezoid(f, th)

MU = {}
for E in ENERGIES:
    mi = NA_e*kn_sigma(E); mc_ = N_mol*rayleigh_sigma(12.398/(E*1000))
    tot = mi+mc_+MU_PAIR_CM+MU_PE_CM
    MU[E] = dict(inc=mi, coh=mc_, pair=MU_PAIR_CM, pe=MU_PE_CM, tot=tot)

def rot_dir(u, v, w, cth, phi):
    # new direction given scattering angle cos cth and azimuth phi relative to (u,v,w)
    sth = np.sqrt(np.maximum(0.0, 1-cth**2))
    # build orthonormal basis
    a = np.where(np.abs(w) < 0.9, 0.0, 1.0)
    b = np.where(np.abs(w) < 0.9, 1.0, 0.0)
    # perpendicular vector e1 = normalize(cross(d, ref))
    rx, ry, rz = a, b, np.zeros_like(a)
    e1x = v*rz - w*ry; e1y = w*rx - u*rz; e1z = u*ry - v*rx
    n1 = np.sqrt(e1x**2+e1y**2+e1z**2)
    e1x/=n1; e1y/=n1; e1z/=n1
    e2x = v*e1z - w*e1y; e2y = w*e1x - u*e1z; e2z = u*e1y - v*e1x
    nx = sth*(np.cos(phi)*e1x + np.sin(phi)*e2x) + cth*u
    ny = sth*(np.cos(phi)*e1y + np.sin(phi)*e2y) + cth*v
    nz = sth*(np.cos(phi)*e1z + np.sin(phi)*e2z) + cth*w
    nn = np.sqrt(nx**2+ny**2+nz**2)
    return nx/nn, ny/nn, nz/nn

def sample_KN(E, n):
    k = E/me
    out = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        m = todo.size
        c = 2*rng.random(m)-1
        eps = 1/(1+k*(1-c))
        f = eps**2*(eps+1/eps-(1-c**2))
        acc = rng.random(m)*2 < f
        out[todo[acc]] = c[acc]
        todo = todo[~acc]
    return out

def sample_rayl_cos(lam_A, n):
    u = (1-rng.random(n))  # avoid trivial
    U = (1-rng.random(n))**(-1/3) - 1  # pdf ∝ 1/(1+u)^4 in u=(s/a)^2 ... see notes
    s = 0.3*np.sqrt(U)
    sh = np.clip(lam_A*s, 0, 1)   # sin(theta/2)
    return 1-2*sh**2

def sample_sauter_like(E, n):
    # photoelectron angle: pdf ∝ sin^2 th/(1-beta cos th)^4, rejection
    g = 1+E/me; beta = np.sqrt(1-1/g**2)
    out = np.empty(n); todo = np.arange(n)
    while todo.size:
        m=todo.size
        c = 2*rng.random(m)-1
        f = (1-c**2)/(1-beta*c)**4
        fmax = 1/(1-beta)**4*1.0
        acc = rng.random(m)*fmax < f
        out[todo[acc]] = c[acc]; todo=todo[~acc]
    return out

def isotropic(n):
    c = 2*rng.random(n)-1; phi = 2*np.pi*rng.random(n)
    s = np.sqrt(1-c**2)
    return s*np.cos(phi), s*np.sin(phi), c

# geometry
ZMAX=30.0; HALF=15.0
SRC_Z = -100.0
Z_SC0, Z_SC1 = 4.9, 5.1
R_SC = 1.0

def in_volume(x,y,z):
    return (x*x+y*y <= R_SC**2) & (z>=Z_SC0) & (z<=Z_SC1)

def run(nprim_per_E):
    events = []  # dict rows
    n_total_emitted = 0
    for E0 in ENERGIES:
        n = nprim_per_E
        # sample source direction: uniform solid angle in pyramid; tan-plane sampling with density (1+x^2+y^2)^(-3/2)
        acc_x=[];acc_y=[]
        need=n
        xs=[];ys=[]
        while need>0:
            m = int(need*1.2)+10
            x = (2*rng.random(m)-1)*0.05
            y = (2*rng.random(m)-1)*0.05
            keep = rng.random(m) < (1+x*x+y*y)**(-1.5)
            xs.append(x[keep]); ys.append(y[keep])
            need -= keep.sum()
        x = np.concatenate(xs)[:n]; y=np.concatenate(ys)[:n]
        d = np.sqrt(1+x*x+y*y)
        u = x/d; v = y/d; w = 1/d
        # start at entrance surface z=0: source at z=-100, scale
        t = (0-SRC_Z)/w
        px = 0 + u*t; py = 0 + v*t; pz = np.zeros(n)
        E = np.full(n, E0)
        n_total_emitted += n
        # active photon arrays
        P = dict(x=px,y=py,z=pz,u=u,v=v,w=w,E=E)
        active_ids = None
        # loop over generations via stack of arrays
        stack=[P]
        while stack:
            P = stack.pop()
            ev = {}
            x=P['x'];y=P['y'];z=P['z'];u=P['u'];v=P['v'];w=P['w'];E=P['E']
            m=x.size
            if m==0: continue
            mt = np.array([MU_tot_of(e) for e in E]) if False else None
            # per-energy mu: arrays
            muinc = np.where(E>0, 0,0.0)
            mu_tot = np.interp(E, ENERGIES_ARR, MU_TOT_ARR)
            s = -np.log(1-rng.random(m))/mu_tot
            xn = x+u*s; yn = y+v*s; zn = z+w*s
            inside = (np.abs(xn)<=HALF)&(np.abs(yn)<=HALF)&(zn>=0)&(zn<=ZMAX)
            # photons that interact inside
            idx = np.where(inside)[0]
            if idx.size==0: continue
            x=xn[idx]; y=yn[idx]; z=zn[idx]; u=u[idx]; v=v[idx]; w=w[idx]; E=E[idx]
            Ei = E
            m=idx.size
            mu_i = np.interp(Ei, ENERGIES_ARR, MU_INC_ARR)
            mu_c = np.interp(Ei, ENERGIES_ARR, MU_COH_ARR)
            mu_p = np.interp(Ei, ENERGIES_ARR, MU_PAIR_ARR)
            mu_e = np.interp(Ei, ENERGIES_ARR, MU_PE_ARR)
            mu_tt = mu_i+mu_c+mu_p+mu_e
            r = rng.random(m)*mu_tt
            proc = np.where(r<mu_i,0, np.where(r<mu_i+mu_c,1, np.where(r<mu_i+mu_c+mu_p,2,3)))
            # Compton
            ic = np.where(proc==0)[0]
            ir = np.where(proc==1)[0]
            ipr = np.where(proc==2)[0]
            ipe = np.where(proc==3)[0]
            rows=[]
            insc = in_volume(x,y,z)
            # --- Compton
            if ic.size:
                Eic=Ei[ic]
                cth = sample_KN_vec(Eic)
                eps = 1/(1+Eic/me*(1-cth))
                Eout = eps*Eic
                phi = 2*np.pi*rng.random(ic.size)
                nu,nv,nw = rot_dir(u[ic],v[ic],w[ic],cth,phi)
                # electron direction
                te = Eic-Eout
                pe_x = Eic*(0) ; # compute cos_electron using momentum balance
                pe_mag = np.sqrt(np.maximum(Eic**2+Eout**2-2*Eic*Eout*cth,1e-30))
                cos_e = (Eic - Eout*cth)/pe_mag
                for j,ii in enumerate(ic):
                    if insc[ii]:
                        rows.append(dict(process='compton', e_in_MeV=float(Eic[j]), e_out_MeV=float(Eout[j]),
                                         cos_scatter=float(cth[j]), t_e_MeV=float(te[j]),
                                         cos_electron=float(np.clip(cos_e[j],-1,1)), t_positron_MeV=None))
                # propagate scattered photons
                stack.append(dict(x=x[ic],y=y[ic],z=z[ic],u=nu,v=nv,w=nw,E=Eout))
            # --- Rayleigh
            if ir.size:
                cth = sample_rayl_vec(Ei[ir])
                phi = 2*np.pi*rng.random(ir.size)
                nu,nv,nw = rot_dir(u[ir],v[ir],w[ir],cth,phi)
                for j,ii in enumerate(ir):
                    if insc[ii]:
                        rows.append(dict(process='rayleigh', e_in_MeV=float(Ei[ii]), e_out_MeV=float(Ei[ii]),
                                         cos_scatter=float(cth[j]), t_e_MeV=0.0, cos_electron=None, t_positron_MeV=None))
                stack.append(dict(x=x[ir],y=y[ir],z=z[ir],u=nu,v=nv,w=nw,E=Ei[ir]))
            # --- Pair
            if ipr.size:
                Ep = Ei[ipr]
                Tt = Ep-2*me
                frac = rng.random(ipr.size)
                te_k = Tt*frac; tp_k = Tt*(1-frac)
                # electron angle ~ m/(T+m) scaled
                ang = np.abs(rng.normal(0,1,ipr.size))*me/(te_k+me)
                cos_el = np.cos(ang)
                for j,ii in enumerate(ipr):
                    if insc[ii]:
                        rows.append(dict(process='pair', e_in_MeV=float(Ep[j]), e_out_MeV=None, cos_scatter=None,
                                         t_e_MeV=float(te_k[j]), cos_electron=float(cos_el[j]), t_positron_MeV=float(tp_k[j])))
                # annihilation photons (two 0.511 MeV) from positron at rest at vertex
                ax,ay,az = isotropic(ipr.size)
                # photon 1 direction a, photon 2 = -a
                a_x=np.concatenate([ax,-ax]); a_y=np.concatenate([ay,-ay]); a_z=np.concatenate([az,-az])
                xs2=np.concatenate([x[ipr],x[ipr]]); ys2=np.concatenate([y[ipr],y[ipr]]); zs2=np.concatenate([z[ipr],z[ipr]])
                stack.append(dict(x=xs2,y=ys2,z=zs2,u=a_x,v=a_y,w=a_z,E=np.full(2*ipr.size,me)))
            # --- Photoelectric
            if ipe.size:
                Ep=Ei[ipe]
                cpe = sample_sauter_like(0,0) if False else None
                cpe = sample_pe_vec(Ep)
                for j,ii in enumerate(ipe):
                    if insc[ii]:
                        rows.append(dict(process='photoelectric', e_in_MeV=float(Ep[j]), e_out_MeV=None, cos_scatter=None,
                                         t_e_MeV=float(Ep[j]), cos_electron=float(cpe[j]), t_positron_MeV=None))
            events.extend(rows)
        
    return events, n_total_emitted

def sample_KN_vec(E):
    out=np.empty(E.size)
    k=E/me
    todo=np.arange(E.size)
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
    g = 1+E/me; beta=np.sqrt(1-1/g**2)
    out=np.empty(E.size); todo=np.arange(E.size)
    while todo.size:
        b=beta[todo]; m=todo.size
        c=2*rng.random(m)-1
        f=(1-c**2)/(1-b*c)**4
        fmax=1/(1-b)**4
        acc=rng.random(m)*fmax<f
        out[todo[acc]]=c[acc]; todo=todo[~acc]
    return out

ENERGIES_ARR = np.array(ENERGIES)
MU_INC_ARR = np.array([MU[E]['inc'] for E in ENERGIES])
MU_COH_ARR = np.array([MU[E]['coh'] for E in ENERGIES])
MU_PAIR_ARR = np.array([MU[E]['pair'] for E in ENERGIES])
MU_PE_ARR = np.array([MU[E]['pe'] for E in ENERGIES])
MU_TOT_ARR = MU_INC_ARR+MU_COH_ARR+MU_PAIR_ARR+MU_PE_ARR
'''
open('mc.py','w').write(mc)
print(subprocess.run(['python','-c','import sys; sys.argv=["x"]; exec(open("mc.py").read()); print(MU)'],capture_output=True,text=True))
