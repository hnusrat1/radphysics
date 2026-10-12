
import numpy as np, time

me = 0.51099895
re = 2.8179403262e-13
NA = 6.02214076e23
n_e_g = NA*10/18.01528

def kn_total(E):
    eta = np.asarray(E,dtype=float)/me
    a = 1.0+2.0*eta
    return 2*np.pi*re**2*((1+eta)/eta**2*(2*(1+eta)/a - np.log(a)/eta) + np.log(a)/(2*eta) - (1+3*eta)/a**2)

Egrid = np.array([0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0,3.0])
muR  = np.array([0.028,0.012,0.0085,0.0060,0.0052,0.0045,0.0033,0.0028,0.0022,0.0019,0.0017,0.0015,0.0012,0.0010,0.0009,0.00085,0.0008,0.00075,0.00055,0.00040])
tau  = np.array([0.60,0.15,0.045,0.0093,0.0026,0.0019,0.00070,0.00042,0.00016,0.000085,0.000052,0.000036,0.000021,0.000013,0.000010,0.000009,0.000008,0.000007,0.000004,0.0000015])
kap  = np.zeros_like(Egrid)
sel = Egrid>1.022
kap[sel]=np.interp(np.clip(Egrid[sel],1.1732,3.0),[1.1732,1.3325,1.5,2.0,3.0],[0.00012,0.00075,0.0011,0.0026,0.0055])

def interp1(x, xp, fp):
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp)
def mu_comp(E): return kn_total(np.clip(E,0.01,3.0))*n_e_g
def mu_ray(E): return interp1(E, Egrid, muR)
def mu_photo(E): return interp1(E, Egrid, tau)
def mu_pair(E):
    E=np.asarray(E,float)
    return np.where(E>2*me, interp1(E, Egrid, kap), 0.0)
def mu_totX(E):
    return mu_comp(E)+mu_ray(E)+mu_photo(E)+mu_pair(E)

# KN quantile table
eta_grid = np.linspace(0.02, 3.0, 400)
xi_grid = np.linspace(0.001, 0.999, 512)
mu_grid = np.linspace(-1, 1, 4001)
def kn_dsdmu_vec(eta, mu):
    r = 1.0/(1.0+eta*(1.0-mu))
    return np.pi*re**2 * r**2 * (r + 1.0/r - (1.0-mu*mu))
Q = np.zeros((len(eta_grid), len(xi_grid)))
for i,eta in enumerate(eta_grid):
    d = kn_dsdmu_vec(eta, mu_grid)
    c = np.cumsum(d); c = c/c[-1]
    Q[i,:] = np.interp(xi_grid, c, mu_grid)

def sample_kn_cos(E, xi):
    eta = np.clip(E/me, eta_grid[0], eta_grid[-1])
    ie = (eta - eta_grid[0])/(eta_grid[-1]-eta_grid[0])*(len(eta_grid)-1)
    i0 = np.clip(np.floor(ie).astype(int),0,len(eta_grid)-2)
    f = ie-i0
    ix = xi*(len(xi_grid)-1)
    j0 = np.clip(np.floor(ix).astype(int),0,len(xi_grid)-2)
    fx = ix-j0
    v0 = Q[i0,j0]*(1-fx)+Q[i0,j0+1]*fx
    v1 = Q[i0+1,j0]*(1-fx)+Q[i0+1,j0+1]*fx
    return v0*(1-f)+v1*f

def rotate(u, cosT, phi):
    ux,uy,uz = u[...,0],u[...,1],u[...,2]
    e1x = uy; e1y = -ux; e1z = np.zeros_like(ux)
    n = np.sqrt(e1x*e1x+e1y*e1y)
    e1x=np.where(n>0,e1x/n,1.0); e1y=np.where(n>0,e1y/n,0.0)
    e2x = uy*e1z-uz*e1y; e2y = uz*e1x-ux*e1z; e2z = ux*e1y-uy*e1x
    st = np.sqrt(np.clip(1-cosT*cosT,0,1))
    cphi = np.cos(phi); sphi = np.sin(phi)
    nx = e1x*(st*cphi)+e2x*(st*sphi)+ux*cosT
    ny = e1y*(st*cphi)+e2y*(st*sphi)+uy*cosT
    nz = e1z*(st*cphi)+e2z*(st*sphi)+uz*cosT
    return np.stack([nx,ny,nz],axis=-1)

def score_cylinder(x,y,z,ux,uy,uz,s, zlo=4.9, zhi=5.1, rad=1.0):
    tz1=(zlo-z)/uz; tz2=(zhi-z)/uz
    tza=np.minimum(tz1,tz2); tzb=np.maximum(tz1,tz2)
    tza=np.clip(tza,0.0,s); tzb=np.clip(tzb,0.0,s)
    uz0 = np.abs(uz)<1e-9
    tza=np.where(uz0, 0.0, tza); tzb=np.where(uz0, s, tzb)
    tza=np.where(uz0 & ((z<zlo)|(z>zhi)), 0.0, tza)
    tzb=np.where(uz0 & ((z<zlo)|(z>zhi)), 0.0, tzb)
    a=ux*ux+uy*uy
    b=2.0*(x*ux+y*uy)
    c=x*x+y*y-rad*rad
    disc=b*b-4*a*c
    a0 = np.abs(a)<1e-9
    sq=np.sqrt(np.clip(disc,0,None))
    den=2.0*a+1e-30
    t1=(-b-sq)/den; t2=(-b+sq)/den
    neg = disc<0
    tlo=np.where(neg, 0.0, np.where(c<0, 0.0, np.minimum(t1,t2)))
    thi=np.where(neg, np.where(c<0, s, 0.0), np.maximum(t1,t2))
    # if starts inside and disc>=0: inside [0,t2]
    start_in = c<0
    tlo=np.where(start_in & ~neg & ~a0, 0.0, tlo)
    thi=np.where(start_in & ~neg & ~a0, np.maximum(t1,t2), thi)
    tlo=np.where(a0, np.where(c<0,0.0,0.0), tlo)
    thi=np.where(a0, np.where(c<0,s,0.0), thi)
    tlo=np.maximum(tlo, tza); thi=np.minimum(thi, tzb)
    return np.maximum(thi-tlo,0.0)

def run(N, seed=42, verbose=True):
    t0=time.time()
    rng = np.random.default_rng(seed)
    x0=rng.uniform(-5,5,N); y0=rng.uniform(-5,5,N)
    d=np.sqrt(x0*x0+y0*y0+1e4)
    ux=x0/d; uy=y0/d; uz=100.0/d
    E=np.where(rng.random(N)<0.5, 1.1732,1.3325)
    x=np.zeros(N); y=np.zeros(N); z=np.zeros(N)
    scat=np.zeros(N,bool)
    alive=np.ones(N,bool)
    nb_ph=28
    ph_bins=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
    ct_bins=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
    Hp = np.zeros((28,10,2))  # [energy bin, cos bin, category(0 prim,1 scat)] accumulated track cm
    Hcnt = np.zeros(2)
    tot_int = 0; n_pair = 0
    pending = []
    it=0
    while True:
        idx = np.where(alive)[0]
        if len(idx)==0:
            if len(pending)==0: break
            arr = np.array(pending); pending=[]
            x=arr[:,0]; y=arr[:,1]; z=arr[:,2]; ux=arr[:,3]; uy=arr[:,4]; uz=arr[:,5]; E=arr[:,6]
            scat=np.ones(len(arr),bool); alive=np.ones(len(arr),bool); idx=np.arange(len(arr))
        else:
            x=x[idx]; y=y[idx]; z=z[idx]; ux=ux[idx]; uy=uy[idx]; uz=uz[idx]; E=E[idx]; scat=scat[idx]
        n=len(x)
        mu = mu_totX(E)
        s = -np.log(rng.random(n))/mu
        # exit distances
        tex = np.full(n, np.inf)
        # z boundaries
        for lim, sign in [(30.0,1.0),(0.0,-1.0)]:
            tz = (lim-z)/uz
            ok = (sign*uz)>0
            tex = np.where(ok, np.minimum(tex, np.where(tz>0,tz,np.inf)), tex)
        for lim, sign in [(15.0,1.0),(-15.0,-1.0)]:
            tx = (lim-x)/ux
            ok = (sign*ux)>0
            tex = np.where(ok, np.minimum(tex, np.where(tx>0,tx,np.inf)), tex)
            ty = (lim-y)/uy
            ok = (sign*uy)>0
            tex = np.where(ok, np.minimum(tex, np.where(ty>0,ty,np.inf)), tex)
        s_eff = np.minimum(s, tex)
        tl = score_cylinder(x,y,z,ux,uy,uz,s_eff)
        # histogram: energy bin index from E, cos bin from uz
        eb = np.clip(np.searchsorted(ph_bins, E, side='right')-1, 0, 27)
        cb = np.clip(np.searchsorted(ct_bins, uz, side='right')-1, 0, 9)
        cat = scat.astype(int)
        np.add.at(Hp, (eb, cb, cat), tl)
        Hcnt[0]+= tl[~scat].sum(); Hcnt[1]+= tl[scat].sum()
        # interactions
        interact = (s < tex)
        nit = interact.sum()
        tot_int += nit
        if nit>0:
            Ei=E[interact]; xi=x[interact]; yi=y[interact]; zi=z[interact]
            uxi=ux[interact]; uyi=uy[interact]; uzi=uz[interact]; sci=scat[interact]
            mui = mu_totX(Ei)
            pc = mu_comp(Ei)/mui
            pr = pc + mu_ray(Ei)/mui
            pp = pr + mu_photo(Ei)/mui
            # pair prob = mu_pair/mu
            r = rng.random(nit)
            # compton
            mC = r < pc
            # rayleigh
            mR = (r>=pc)&(r<pr)
            mPh = (r>=pr)&(r<pp)
            mPa = (r>=pp)
            # update positions of all interacting
            na=nit
            xnew = x; ynew=y; znew=z; uxnew=ux; uynew=uy; uznew=uz; Enew=E; scatnew=scat
            xnew[interact] = xi + uxi*s[interact]
            ynew[interact] = yi + uyi*s[interact]
            znew[interact] = zi + uzi*s[interact]
            # compton scatter
            if mC.any():
                xi_r = rng.random(mC.sum())
                cosT = sample_kn_cos(Ei[mC], xi_r)
                phi = 2*np.pi*rng.random(mC.sum())
                u = np.stack([uxi[mC],uyi[mC],uzi[mC]],axis=-1)
                un = rotate(u, cosT, phi)
                ind = np.where(interact)[0][mC]
                uxnew[ind]=un[:,0]; uynew[ind]=un[:,1]; uznew[ind]=un[:,2]
                Enew[ind] = Ei[mC]/(1.0+(Ei[mC]/me)*(1.0-cosT))
                scatnew[ind]=True
            # rayleigh: rotate tiny angle
            if mR.any():
                cosT = 1.0 - 2e-5*rng.random(mR.sum())
                phi = 2*np.pi*rng.random(mR.sum())
                u = np.stack([uxi[mR],uyi[mR],uzi[mR]],axis=-1)
                un = rotate(u, cosT, phi)
                ind = np.where(interact)[0][mR]
                uxnew[ind]=un[:,0]; uynew[ind]=un[:,1]; uznew[ind]=un[:,2]
                scatnew[ind]=True
            # photoelectric: dead
            if mPh.any():
                ind = np.where(interact)[0][mPh]
                uxnew[ind]=0; uynew[ind]=0; uznew[ind]=0; Enew[ind]=0
            # pair: dead, add two 0.511 photons
            if mPa.any():
                npa = mPa.sum()
                n_pair += npa
                ind = np.where(interact)[0][mPa]
                uxnew[ind]=0; uynew[ind]=0; uznew[ind]=0; Enew[ind]=0
                # two photons, random opposite directions
                th = rng.random(npa)*np.pi
                ph = rng.random(npa)*2*np.pi
                d1 = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)],axis=-1)
                d2 = -d1
                for dd in (d1,d2):
                    pending.append(np.column_stack([xi[mPa],yi[mPa],zi[mPa],dd[:,0],dd[:,1],dd[:,2],np.full(npa,0.511)]))
            # low energy cutoff
            alive_new = alive.copy()
            alive_new[interact] = Enew[interact] > 0.01
            alive_new[interact] &= (uxnew[interact]**2+uynew[interact]**2+uznew[interact]**2)>0.99
            # exited: dead
            alive_new[interact] = alive_new[interact] & ~(s>=tex)[interact]
            # update state arrays for alive subset
            alive=alive_new
        else:
            # all exited (no interactions) -> they die
            alive[interact] = False
            alive[:] = False
        it+=1
        if verbose and it%50==0:
            print('iter',it,'alive',alive.sum(),'pending',len(pending),'tot_int',tot_int,'time',time.time()-t0)
    Vcyl = np.pi*1.0*0.2
    # fluence per phi0: track/Vcyl * 110.25
    f_prim = Hcnt[0]/Vcyl*110.25
    f_scat = Hcnt[1]/Vcyl*110.25
    # spectrum per phi0 per bin (fluence per bin): Hp sum over cos and cat
    sp_prim = Hp[:,:,0].sum(axis=1)/Vcyl*110.25
    sp_scat = Hp[:,:,1].sum(axis=1)/Vcyl*110.25
    ang_prim = Hp[:,:,0].sum(axis=0)/Hcnt[0]
    ang_scat = Hp[:,:,1].sum(axis=0)/Hcnt[1]
    print('primary fluence/Phi0',f_prim,'scattered',f_scat,'total',f_prim+f_scat,'scat frac',f_scat/(f_prim+f_scat))
    print('ang prim:',np.round(ang_prim,4))
    print('ang scat:',np.round(ang_scat,4))
    print('n events',tot_int,'n_pair',n_pair,'time',time.time()-t0)
    return dict(f_prim=f_prim,f_scat=f_scat,sp_prim=sp_prim,sp_scat=sp_scat,ang_prim=ang_prim,ang_scat=ang_scat,Hp=Hp,Hcnt=Hcnt)

if __name__=='__main__':
    r = run(1000000, seed=42, verbose=True)
