code = r'''
import numpy as np, time

estar = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
eT = estar[:,0]; Scol = estar[:,1]; Srad = estar[:,2]; Yy = estar[:,5]
lnT = np.log(eT)
def scol_at(T):
    return np.exp(np.interp(np.log(np.clip(T,1e-4,20.)), lnT, np.log(Scol)))
def srad_at(T):
    return np.exp(np.interp(np.log(np.clip(T,1e-4,20.)), lnT, np.log(Srad)))

m_e=0.5109989461
X0=36.08
Es=21.2
TCUT=0.010
n_e=3.343e23
r02=2.8179403227e-13**2

def moller_rate(T):
    beta2 = 1.0 - 1.0/((T/m_e+1.0)**2)
    gamma = 1.0+T/m_e
    C = 2*np.pi*r02*m_e/np.maximum(beta2,1e-6)
    Tc=TCUT
    bracket = 1.0/Tc - 1.0/(T-Tc) + (2*gamma-1)/(gamma*gamma*T)*np.log(Tc/(T-Tc))
    return n_e*C*bracket

def sample_delta_energy(T_arr, rng):
    T_arr=np.asarray(T_arr,float)
    n=len(T_arr)
    gamma=1.0+T_arr/m_e
    beta2=1.0-1.0/(gamma*gamma)
    C=2*np.pi*r02*m_e/np.maximum(beta2,1e-6)
    Tc=TCUT
    U=(T_arr-Tc)/2.0
    a=1.0/Tc; b=1.0/U
    out=np.empty(n)
    rem=np.arange(n)
    for it in range(30):
        if len(rem)==0: break
        nb=len(rem)
        u=rng.random(nb)
        Ts=1.0/(a - u*(a-b[rem]))
        dsd=C[rem]*(1.0/Ts**2 + 1.0/(T_arr[rem]-Ts)**2 - (2*gamma[rem]-1)/(gamma[rem]**2*Ts*(T_arr[rem]-Ts)))
        env=2.0*C[rem]/Ts**2
        acc=rng.random(nb) < dsd/env
        out[rem[acc]]=Ts[acc]
        rem=rem[~acc]
    if len(rem)>0:
        u=rng.random(len(rem))
        out[rem]=Tc*np.exp(u*np.log(U[rem]/Tc))
    return out

def rot_dir(u, theta, phi):
    a = np.column_stack([-u[:,1], u[:,0], np.zeros(len(u))])
    na=np.sqrt(a[:,0]**2+a[:,1]**2)
    bad=na<1e-8
    a[bad]=[1,0,0]
    na=np.sqrt(a[:,0]**2+a[:,1]**2+a[:,2]**2)
    a=a/np.maximum(na,1e-30)[:,None]
    b=np.cross(u,a)
    ct=np.cos(theta); st=np.sin(theta)
    return u*ct[:,None] + st[:,None]*(a*np.cos(phi)[:,None]+b*np.sin(phi)[:,None])

def electron_run(src_pos, src_dir, src_T, src_par, src_kind, weights, depths, seed=1,
                 maxsteps=500, verbose=True):
    rng=np.random.default_rng(seed)
    depths=np.asarray(depths,float)
    nsc=len(depths)
    dose=np.zeros(nsc); dose_scat=np.zeros(nsc)
    i5=int(np.argmin(np.abs(depths-5.0)))
    flu_tot5=0.0; flu_delta5=0.0; flu_e_spectrum=np.zeros(25); flu_angular=np.zeros(10)
    rad_col_num=0.0; rad_col_den=0.0
    e_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.4])
    cos_edges=np.linspace(-1,1,11)
    queue=[(src_pos.astype(np.float64), src_dir.astype(np.float64), src_T.astype(np.float64),
            src_par.astype(np.int8), src_kind.astype(np.int8), weights.astype(np.float64),
            np.zeros(len(src_T),dtype=bool))]
    n_batches=0; total_steps=0
    t0=time.time()
    while queue:
        cp,cd,cT,cpar,ckind,cw,cdl = queue.pop(0)
        n0=len(cT)
        if n0==0: continue
        n_batches+=1
        step=0
        while len(cT)>0 and step<maxsteps:
            step+=1; total_steps+=len(cT)
            Sc=scol_at(cT); Sr=srad_at(cT)
            ds=np.minimum(0.02, 0.08*cT/np.maximum(Sc,1e-6))
            dEc=Sc*ds; dEr=Sr*ds
            # delta production
            mask_d = cT>0.05
            if mask_d.any():
                rate=moller_rate(cT[mask_d])
                p_pois=np.maximum(rate*ds[mask_d],0.0)
                nd=rng.poisson(p_pois)
                if nd.sum()>0:
                    reps=np.repeat(np.arange(mask_d.sum()), nd)
                    Ts=sample_delta_energy(cT[mask_d][reps], rng)
                    cosf=np.sqrt(np.maximum(Ts*(cT[mask_d][reps]+2*m_e)/(cT[mask_d][reps]*(Ts+2*m_e)),0))
                    cosf=np.minimum(cosf,1.0)
                    phid=rng.uniform(0,2*np.pi,len(Ts))
                    th=np.arccos(cosf)
                    newd=rot_dir(cd[mask_d][reps], th, phid)
                    queue.append((cp[mask_d][reps], newd, Ts, cpar[mask_d][reps], ckind[mask_d][reps],
                                  cw[mask_d][reps], np.ones(len(Ts),dtype=bool)))
            # multiple scattering rotation
            pc=np.sqrt(np.maximum(cT*(cT+2*m_e),0)); beta=pc/(cT+m_e)
            theta_sig=np.sqrt(2.0)*Es/np.maximum(beta*pc,1e-6)*np.sqrt(ds/X0)
            theta_sig=np.minimum(theta_sig,2.0)
            th2=rng.rayleigh(theta_sig)
            phi2=rng.uniform(0,2*np.pi,len(cT))
            newd=rot_dir(cd, th2, phi2)
            # advance
            dEc2=dEc; dEr2=dEr
            pmid = cp + newd*ds[:,None]*0.5
            pnew = cp + newd*ds[:,None]
            Tnew = cT - (dEc2+dEr2)
            # scoring
            r2=pmid[:,0]**2+pmid[:,1]**2
            for kd,zd in enumerate(depths):
                mask=(np.abs(pmid[:,2]-zd)<=0.1)&(r2<=1.0)
                if mask.any():
                    add=dEc2[mask]*cw[mask]
                    dose[kd]+=add.sum()
                    dose_scat[kd]+=add[ cpar[mask]==0 ].sum()
            m5=(np.abs(pmid[:,2]-depths[i5])<=0.1)&(r2<=1.0)
            if m5.any():
                flu_tot5 += (ds[m5]*cw[m5]).sum()
                flu_delta5 += (ds[m5]*cdl[m5]*cw[m5]).sum()
                bin_e=np.clip(np.searchsorted(e_edges, cT[m5],'right')-1,0,24)
                np.add.at(flu_e_spectrum, bin_e, ds[m5]*cw[m5])
                bin_c=np.clip(np.searchsorted(cos_edges, newd[m5][:,2],'right')-1,0,9)
                np.add.at(flu_angular, bin_c, ds[m5]*cw[m5])
                rad_col_num += (Sr[m5]*ds[m5]*cw[m5]).sum()
                rad_col_den += (Sc[m5]*ds[m5]*cw[m5]).sum()
            # kill
            rnew=np.sqrt(pnew[:,0]**2+pnew[:,1]**2)
            dead=(Tnew<TCUT)|(rnew>5.5)|(pnew[:,2]<-1.5)|(pnew[:,2]>15.0)
            keep=~dead
            cp=pnew[keep]; cd=newd[keep]; cT=Tnew[keep]; cpar=cpar[keep]
            ckind=ckind[keep]; cw=cw[keep]; cdl=cdl[keep]
        if verbose and n_batches%5==0:
            print(f'  batch {n_batches} done, queue {len(queue)}, elapsed {time.time()-t0:.1f}s')
    if verbose:
        print(f'electron_run done: batches={n_batches}, total_steps={total_steps}, elapsed {time.time()-t0:.1f}s')
    return dict(dose=dose, dose_scat=dose_scat, flu_tot5=flu_tot5, flu_delta5=flu_delta5,
                flu_e_spectrum=flu_e_spectrum, flu_angular=flu_angular,
                rad_col_num=rad_col_num, rad_col_den=rad_col_den, i5=i5)
'''
open('/tmp/electronmc.py','w').write(code)
print('written')