code = r'''
import numpy as np, time

estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
eT = estar[:,0]; Scol = estar[:,1]; Srad = estar[:,2]; Yy = estar[:,5]
lnT = np.log(eT)
def scol_at(T):
    return np.exp(np.interp(np.log(np.clip(T,1e-4,20.)), lnT, np.log(Scol)))
def srad_at(T):
    return np.exp(np.interp(np.log(np.clip(T,1e-4,20.)), lnT, np.log(Srad)))
def yield_at(T):
    return np.interp(T, eT, Yy)

m_e=0.5109989461
X0=36.08
Es=21.2
TCUT=0.010
n_e=3.343e23  # electrons/cm3 water
r02=2.8179403227e-13**2

def moller_rate(T):
    # dN/ds above TCUT
    beta2 = 1.0 - 1.0/((T/m_e+1.0)**2)
    gamma = 1.0+T/m_e
    C = 2*np.pi*r02*m_e*np.maximum(beta2,1e-6)**(-1)
    Tc=TCUT
    U = (T-Tc)/2.0
    bracket = 1.0/Tc - 1.0/(T-Tc) + (2*gamma-1)/(gamma*gamma*T)*np.log(Tc/(T-Tc))
    return n_e*C*bracket

def sample_delta_energy(T, n, rng):
    # returns T_s in [TCUT,(T-TCUT)/2], rejection
    gamma=1.0+T/m_e
    beta2=1.0-1.0/(gamma*gamma)
    C=2*np.pi*r02*m_e/np.maximum(beta2,1e-6)
    Tc=TCUT; U=(T-Tc)/2.0
    out=np.zeros(n)
    # proposal sample from 1/Ts^2 on [Tc,U]
    a=1.0/Tc; b=1.0/U
    for it in range(3):
        u=rng.random(n)
        Ts=1.0/(a - u*(a-b))
        # pdf prop ~ 1/Ts^2, exact CDF: g(Ts)= (1/Tc -1/Ts)/(1/Tc-1/U)*(1/Ts^2)... standard
        # q(Ts)=c/Ts^2 with c=(1/Tc-1/U)^-1
        dsdT = C*(1.0/Ts**2 + 1.0/(T-Ts)**2 - (2*gamma-1)/(gamma*gamma*Ts*(T-Ts)))
        # envelope: 2*C/Ts^2
        env = 2.0*C/Ts**2
        acc = rng.random(n) < dsdT/env
        if it==0:
            out = Ts
            remaining = ~acc
        else:
            out[remaining] = Ts[remaining]
            remaining[remaining] = ~acc[remaining]
        if not remaining.any(): break
    # fallback: uniform in log
    rem = remaining if 'remaining' in dir() else np.zeros(n,dtype=bool)
    if rem.any():
        out[rem] = Tc*np.exp(rng.uniform(0, np.log(U/Tc), rem.sum()))
    return out

def rot_dir(u, theta, phi):
    # rotate unit vectors u by polar angle theta, azimuth phi
    a = np.column_stack([-u[:,1], u[:,0], np.zeros(len(u))])
    na=np.sqrt(a[:,0]**2+a[:,1]**2+1e-30)
    bad=na<1e-6
    a[bad]=[1,0,0]
    na=np.sqrt(a[:,0]**2+a[:,1]**2+a[:,2]**2)
    a=a/na[:,None]
    b=np.cross(u,a)
    ct=np.cos(theta); st=np.sin(theta)
    return u*ct[:,None] + st[:,None]*(a*np.cos(phi)[:,None]+b*np.sin(phi)[:,None])

def electron_run(src_pos, src_dir, src_T, src_par, src_kind, weights, depths, seed=1,
                 maxsteps=400, rho=1.0, verbose=True):
    # depths: array of cylinder centers (radius 1, half-thickness 0.1)
    rng=np.random.default_rng(seed)
    depths=np.asarray(depths,float)
    nsc=len(depths)
    dose=np.zeros(nsc); dose_scat=np.zeros(nsc)
    # tallies at 5-cm cylinder: index of depth closest to 5.0
    i5=int(np.argmin(np.abs(depths-5.0)))
    flu_tot5=0.0; flu_delta5=0.0; flu_e_spectrum=np.zeros(25); flu_angular=np.zeros(10)
    rad_col_num=0.0; rad_col_den=0.0
    e_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.4])
    cos_edges=np.linspace(-1,1,11)
    # queue
    queue=[(src_pos.astype(np.float64), src_dir.astype(np.float64), src_T.astype(np.float64),
            src_par.astype(np.int8), src_kind.astype(np.int8), weights.astype(np.float64),
            np.zeros(len(src_T),dtype=bool))]  # last: is_delta
    batch_count=0
    t0=time.time()
    while queue:
        pos, dirv, T, par, kind, w, is_delta = queue.pop(0)
        n=len(T)
        if n==0: continue
        alive=np.ones(n,dtype=bool)
        step=0
        while alive.any() and step<maxsteps:
            step+=1
            n_a=alive.sum()
            p=pos[alive]; d=dirv[alive]; Te=T[alive]; pa=par[alive]; ki=kind[alive]; ww=w[alive]; dl=is_delta[alive]
            Sc=scol_at(Te); Sr=srad_at(Te)
            ds=np.minimum(0.02, 0.08*Te/np.maximum(Sc,1e-6))
            dEc=Sc*ds; dEr=Sr*ds
            # delta production
            if (Te>0.05).any():
                rate=moller_rate(Te)
                lam=np.where(Te>0.05, rate*ds, 0.0)
                nd=rng.poisson(lam)
                if nd.sum()>0:
                    totd=int(nd.sum())
                    # sample energies
                    reps=np.repeat(np.arange(n_a), nd)
                    Ts=sample_delta_energy(Te[reps], totd, rng)
                    # direction
                    cosf=np.sqrt(Ts*(Te[reps]+2*m_e)/(Te[reps]*(Ts+2*m_e)))
                    sinf=np.sqrt(np.maximum(1-cosf*cosf,0))
                    phid=rng.uniform(0,2*np.pi,totd)
                    dird=np.column_stack([sinf*np.cos(phid), sinf*np.sin(phid), cosf])
                    # rotate relative to primary direction: use rot_dir
                    newd=rot_dir(d[reps], np.arccos(np.clip(cosf,-1,1)), phid)
                    # store deltas (parent type inherited, kind inherited, not delta->is_delta True)
                    queue.append((p[reps], newd, Ts.astype(np.float32), pa[reps].astype(np.int8),
                                  ki[reps].astype(np.int8), ww[reps].astype(np.float64),
                                  np.ones(totd,dtype=bool)))
            # multiple scattering
            pc=np.sqrt(np.maximum(Te*(Te+2*m_e),0)); beta=pc/(Te+m_e)
            theta_sig=np.sqrt(2.0)*Es/np.maximum(beta*pc,1e-6)*np.sqrt(ds/X0)
            theta_sig=np.minimum(theta_sig, 2.0)
            th=rng.rayleigh(theta_sig)
            phi=rng.uniform(0,2*np.pi,n_a)
            newd=rot_dir(d, th, phi)
            # advance
            pmid = p + newd*ds[:,None]*0.5
            pnew = p + newd*ds[:,None]
            Tnew = Te - (dEc+dEr)
            # scoring inside cylinders (depth list)
            r2=pmid[:,0]**2+pmid[:,1]**2
            for kd,zd in enumerate(depths):
                mask=(np.abs(pmid[:,2]-zd)<=0.1)&(r2<=1.0)
                if mask.any():
                    add=dEc[mask]*ww[mask]
                    dose[kd]+=add.sum()
                    if pa[mask].sum()==0:
                        # parent scattered flag: par==0 means scattered parent (we stored 1=primary)
                        dose_scat[kd]+=add.sum()
            # 5-cm tallies
            m5=(np.abs(pmid[:,2]-depths[i5])<=0.1)&(r2<=1.0)
            if m5.any():
                flu_tot5 += (ds[m5]*ww[m5]).sum()
                flu_delta5 += (ds[m5]*dl[m5]*ww[m5]).sum()
                bin_e=np.clip(np.searchsorted(e_edges, Te[m5],'right')-1,0,24)
                np.add.at(flu_e_spectrum, bin_e, ds[m5]*ww[m5])
                bin_c=np.clip(np.searchsorted(cos_edges, newd[m5][:,2],'right')-1,0,9)
                np.add.at(flu_angular, bin_c, ds[m5]*ww[m5])
                rad_col_num += (Sr[m5]*ds[m5]*ww[m5]).sum()
                rad_col_den += (Sc[m5]*ds[m5]*ww[m5]).sum()
            # update positions/directions/energies
            # kill: T<TCUT, out of box
            rnew=np.sqrt(pnew[:,0]**2+pnew[:,1]**2)
            dead=(Tnew<TCUT)|(rnew>5.5)|(pnew[:,2]<-1.5)|(pnew[:,2]>15.0)
            alive[alive]=~dead
            pos_new=pnew[~dead]; dir_new=newd[~dead]; T_new=Tnew[~dead]
            # update arrays for alive subset: need new arrays
            # easiest: rebuild
            pos=pos_new if pos_new.shape[0]>0 else np.zeros((0,3))
            dirv=dir_new; T=T_new
            # primitive: reconstruct full arrays? This is slow. Better update in place:
            # We'll handle by direct assignment below instead
            break
        # NOTE: this simple loop needs rework - T/dir arrays must persist across iterations.
        break
    print('electron_run placeholder')
    return None
'''
open('/tmp/electronmc_placeholder.py','w').write(code)
print('written placeholder (needs rework)')