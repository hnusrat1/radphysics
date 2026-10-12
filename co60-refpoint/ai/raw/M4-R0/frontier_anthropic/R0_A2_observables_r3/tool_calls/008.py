
mc = r'''
import numpy as np, phys
from phys import *

SSD=100.0; DIST=105.0
TANH=0.05
A_TAN = (2*TANH)**2
OMEGA = 4*np.arcsin(TANH*TANH/np.sqrt((1+TANH*TANH)**2))
R_CYL=1.0; V_POI = np.pi*R_CYL**2*0.2
LINES = np.array([1.1732, 1.3325])
BOX = (-15.,15.,-15.,15.,0.,30.)
E_CUT = 0.001

# bins
PE_EDGES = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
EE_EDGES = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
CO_EDGES = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])

# depth-dose slabs (cylinder R=1, 0.25 cm bins)
DZ_CENT = np.arange(1,41)*0.25
def run_batch(N, rng, depth_bins=True):
    """Transport N scattered-candidate histories. Returns dict of tallies (per-history sums)."""
    # sample directions
    u = (2*rng.random(N)-1)*TANH; v = (2*rng.random(N)-1)*TANH
    w = (A_TAN/OMEGA)*(1+u*u+v*v)**(-1.5)
    dn = np.sqrt(u*u+v*v+1)
    dx,dy,dz = u/dn, v/dn, 1/dn
    t0 = SSD/dz
    px = dx*t0; py = dy*t0; pz = np.zeros(N)+0.0
    pz = dz*t0 - SSD
    # now pz = 0 at surface: check
    px = dx*t0; py = dy*t0; pz = dz*t0 - SSD
    E = np.where(rng.random(N)<0.5, LINES[0], LINES[1])
    flag = np.zeros(N, bool)   # True once interacted (scattered)
    # tallies
    Tsc = np.zeros(PE_EDGES.size-1)     # track in V by photon energy bin (scattered)
    Tco = np.zeros(CO_EDGES.size-1)     # by direction cos bin
    Tprim = 0.0
    cnt = {'coh':0.,'inc':0.,'pe':0.,'pair':0.}
    etr = {'coh':0.,'inc':0.,'pe':0.,'pair':0.}   # energy transfer weighted (sum of count*<T>)
    cnt_p = {'coh':0.,'inc':0.,'pe':0.,'pair':0.}
    Esum_w = 0.0; EwL = 0.0
    # electron births (scattered parents): list of arrays
    birth_T=[]; birth_w=[]; birth_proc=[]; birth_cos=[]; birth_sign=[]
    # depth-dose tallies (scattered): kerma-like sum w*l*E*mutr per depth bin
    Kdep = np.zeros(DZ_CENT.size)
    Kdep_pri = np.zeros(DZ_CENT.size)
    Tdep = np.zeros(DZ_CENT.size)
    # active arrays
    ids = np.arange(N)
    Wt = w.copy()
    Ka = {}
    # state
    st = dict(px=px,py=py,pz=pz,dx=dx,dy=dy,dz=dz,E=E,w=w,flag=flag)
    # annihilation etc appended through lists
    nsteps=0
    while st['E'].size>0:
        nsteps+=1
        px_,py_,pz_ = st['px'],st['py'],st['pz']
        dx_,dy_,dz_ = st['dx'],st['dy'],st['dz']
        E_ = st['E']; w_ = st['w']; f_ = st['flag']
        mt = mu_tot(E_)
        s = -np.log(rng.random(E_.size))/mt
        # exit distance
        tx = np.where(dx_>0,(BOX[1]-px_)/np.where(dx_>0,dx_,1), np.where(dx_<0,(BOX[0]-px_)/np.where(dx_<0,dx_,1),np.inf))
        ty = np.where(dy_>0,(BOX[3]-py_)/np.where(dy_>0,dy_,1), np.where(dy_<0,(BOX[2]-py_)/np.where(dy_<0,dy_,1),np.inf))
        tz = np.where(dz_>0,(BOX[5]-pz_)/np.where(dz_>0,dz_,1), np.where(dz_<0,(BOX[4]-pz_)/np.where(dz_<0,dz_,1),np.inf))
        texit = np.minimum(np.minimum(tx,ty),tz)
        texit = np.maximum(texit,0.0)
        step = np.minimum(s, texit)
        inter = s < texit
        # score track in POI cylinder for scattered photons
        l_poi = overlap(px_,py_,pz_,dx_,dy_,dz_,step,4.9,5.1,R_CYL)
        m = f_ & (l_poi>0)
        if np.any(m):
            ee = E_[m]; ww = w_[m]; ll = l_poi[m]; cz = dz_[m]
            Tsc += np.histogram(ee, bins=PE_EDGES, weights=ww*ll)[0]
            Tco += np.histogram(cz, bins=CO_EDGES, weights=ww*ll)[0]
            # interaction probabilities and births (expected counts per segment)
            c_coh = mu_coh(ee); c_inc=mu_inc(ee); c_pe=mu_pe(ee); c_pa=mu_pair(ee)
            pref = OMEGA*DIST**2/N   # per Phi0 normalisation of track length (T_V)
            # counts per process (per Phi0, times track length): T_V = pref*sum(w l)
            ww_l = ww*ll*pref
            cnt['coh'] += np.sum(ww_l*c_coh); cnt['inc'] += np.sum(ww_l*c_inc)
            cnt['pe'] += np.sum(ww_l*c_pe); cnt['pair'] += np.sum(ww_l*c_pa)
            # energy transfer
            etr['inc'] += np.sum(ww_l*c_inc*ee*(1-eps_mean(ee)))
            etr['pe'] += np.sum(ww_l*c_pe*ee)
            etr['pair'] += np.sum(ww_l*c_pa*np.maximum(ee-1.022,0))
            Esum_w += np.sum(ww_l*ee); 
            # electron births (sampled)
            # Compton
            eps_, cth_ = kn_sample(ee, rng)
            Tc = ee*(1-eps_)
            # direction of electron
            dd = np.stack([dx_[m],dy_[m],dz_[m]],1)
            phi = 2*np.pi*rng.random(ee.size)
            dsc = rotate_dir(dd, cth_, phi)
            pe_vec = dd - eps_[:,None]*dsc
            pe_vec /= np.linalg.norm(pe_vec,axis=1)[:,None]
            birth_T.append(Tc); birth_w.append(ww_l*c_inc); birth_proc.append(np.zeros(ee.size,int)); birth_cos.append(pe_vec[:,2]); birth_sign.append(np.zeros(ee.size,int))
            # photoelectric
            birth_T.append(ee); birth_w.append(ww_l*c_pe); birth_proc.append(np.ones(ee.size,int)); birth_cos.append(dz_[m]); birth_sign.append(np.zeros(ee.size,int))
            # pair: e- and e+ with uniform sharing
            Tp = np.maximum(ee-1.022,0)
            fr = rng.random(ee.size)
            birth_T.append(Tp*fr); birth_w.append(ww_l*c_pa); birth_proc.append(np.full(ee.size,2)); birth_cos.append(dz_[m]); birth_sign.append(np.zeros(ee.size,int))
            birth_T.append(Tp*(1-fr)); birth_w.append(ww_l*c_pa); birth_proc.append(np.full(ee.size,2)); birth_cos.append(dz_[m]); birth_sign.append(np.ones(ee.size,int))
            # depth-dose: scattered, E*mutr at each depth
            mtr = c_inc*(1-eps_mean(ee)) + c_pe + c_pa*np.maximum(1-1.022/ee,0)
            # (energy-transfer based kerma per track length)
        # depth-dose (slabs) for scattered photon segments
        if depth_bins and np.any(f_):
            mm = f_
            ee_all = E_[mm]; ww_all = w_[mm]
            mtr_all = mu_inc(ee_all)*(1-eps_mean(ee_all)) + mu_pe(ee_all) + mu_pair(ee_all)*np.maximum(1-1.022/ee_all,0)
            for k,zc in enumerate(DZ_CENT):
                lk = overlap(px_[mm],py_[mm],pz_[mm],dx_[mm],dy_[mm],dz_[mm],step[mm],zc-0.125,zc+0.125,R_CYL)
                Kdep[k] += np.sum(ww_all*lk*ee_all*mtr_all)
                Tdep[k] += np.sum(ww_all*lk)
        # move
        npx = px_+step*dx_; npy = py_+step*dy_; npz = pz_+step*dz_
        # alive photons: those that interacted inside box
        alive = inter.copy()
        # process selection for interacting photons
        idx = np.nonzero(alive)[0]
        newlist = []
        if idx.size:
            Ei = E_[idx]
            mi = mt[idx]
            c_coh = mu_coh(Ei); c_inc=mu_inc(Ei); c_pe=mu_pe(Ei); c_pa=mu_pair(Ei)
            xi = rng.random(idx.size)*mi
            is_coh = xi < c_coh
            is_inc = (~is_coh)&(xi < c_coh+c_inc)
            is_pe = (~is_coh)&(~is_inc)&(xi < c_coh+c_inc+c_pe)
            is_pa = (~is_coh)&(~is_inc)&(~is_pe)
            # coherent
            ic = idx[is_coh]
            if ic.size:
                c = rayleigh_cos(ic.size, rng)
                phi = 2*np.pi*rng.random(ic.size)
                dd = np.stack([dx_[ic],dy_[ic],dz_[ic]],1)
                nd = rotate_dir(dd, c, phi)
                dx_[ic]=nd[:,0]; dy_[ic]=nd[:,1]; dz_[ic]=nd[:,2]
                f_[ic]=True
            # incoherent (Compton)
            ii = idx[is_inc]
            if ii.size:
                eps_, cth_ = kn_sample(E_[ii], rng)
                phi = 2*np.pi*rng.random(ii.size)
                dd = np.stack([dx_[ii],dy_[ii],dz_[ii]],1)
                nd = rotate_dir(dd, cth_, phi)
                dx_[ii]=nd[:,0]; dy_[ii]=nd[:,1]; dz_[ii]=nd[:,2]
                E_[ii] = E_[ii]*eps_
                f_[ii]=True
            # photoelectric: absorbed
            ip = idx[is_pe]
            # pair: absorbed, two annihilation photons
            ipa = idx[is_pa]
            # kill: pe and pair photons and those with E < cut
            kill = np.zeros(E_.size,bool)
            kill[ip]=True; kill[ipa]=True
            if ipa.size:
                # annihilation photons: 2 x 0.511, isotropic, scattered
                n2 = ipa.size
                for _ in range(2):
                    c = 2*rng.random(n2)-1; ph = 2*np.pi*rng.random(n2)
                    s_ = np.sqrt(1-c*c)
                    newlist.append(dict(px=npx[ipa],py=npy[ipa],pz=npz[ipa],dx=s_*np.cos(ph),dy=s_*np.sin(ph),dz=c,
                                        E=np.full(n2,M_E),w=w_[ipa],flag=np.ones(n2,bool)))
            # update state for interacting photons
            # (photons killed are removed)
            keep = alive & (~kill)
            keep_idx = np.nonzero(keep)[0]
            st_new = dict(px=npx[keep_idx], py=npy[keep_idx], pz=npz[keep_idx],
                          dx=dx_[keep_idx], dy=dy_[keep_idx], dz=dz_[keep_idx],
                          E=E_[keep_idx], w=w_[keep_idx], flag=f_[keep_idx])
        else:
            st_new = None
        if st_new is None:
            st_new = dict(px=np.zeros(0),py=np.zeros(0),pz=np.zeros(0),dx=np.zeros(0),dy=np.zeros(0),dz=np.zeros(0),E=np.zeros(0),w=np.zeros(0),flag=np.zeros(0,bool))
        # photons that did not interact and remain inside box: continue
        nonint = (~inter)
        # non-interacting photons exit the box (since step==texit) -> dropped
        # low energy cut
        good = st_new['E'] > E_CUT
        st = {k: np.concatenate([st_new[k][good]] + [nl[k] for nl in newlist]) if newlist else st_new[k][good] for k in st_new}
        # (annihilation photons' energy >= cut)
    out = dict(Tsc=Tsc, Tco=Tco, cnt=cnt, etr=etr, Esum_w=Esum_w, Kdep=Kdep, Tdep=Tdep, nsteps=nsteps,
               birth_T=np.concatenate(birth_T), birth_w=np.concatenate(birth_w), birth_proc=np.concatenate(birth_proc),
               birth_cos=np.concatenate(birth_cos), birth_sign=np.concatenate(birth_sign))
    return out
'''
open('/tmp/work/mc.py','w').write(mc)
import mc, importlib
importlib.reload(mc)
import time
rng=np.random.default_rng(1)
t=time.time()
o=mc.run_batch(200000, rng)
print(time.time()-t, o['nsteps'], o['Tsc'].sum()*mc.OMEGA*mc.DIST**2/200000/mc.V_POI)
