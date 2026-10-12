
import numpy as np, time, json, os
from scipy.integrate import quad
from scipy.interpolate import interp1d

r_e = 2.8179403262e-13; r_e2 = r_e*r_e; mec2 = 0.5109989461
NA = 6.02214076e23; M_w = 18.01528
n_mol = NA/M_w; n_e = 10*n_mol; TWOPI = 2*np.pi

ph_edges = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
el_edges = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
ct_edges = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
nph = len(ph_edges)-1; nel = len(el_edges)-1; nct = len(ct_edges)-1

def kn_dsig_dOm(E, costh):
    k = E/mec2
    Ep = E/(1.0 + k*(1.0-costh))
    r = Ep/E
    return 0.5*r_e2*r*r*(1.0/r + r - (1.0 - costh*costh))
def kn_total(E):
    f = lambda th: kn_dsig_dOm(E, np.cos(th))*np.sin(th)
    v,_ = quad(f, 0, np.pi, epsabs=1e-30, limit=400)
    return TWOPI*v

def ray_dsig_dOm_shape(E, costh):
    hcA = 0.01973269804
    q = E*np.sqrt(2*(1-costh))/hcA
    aO, aH = 0.25, 0.20
    FO = 8.0*np.exp(-(q*q*aO*aO)/2.0)
    FH = 1.0*np.exp(-(q*q*aH*aH)/2.0)
    F2 = FO*FO + 2.0*FH*FH
    return 0.5*r_e2*(1.0+costh*costh)*F2

def mu_photo_f(E):
    A = 0.002*(0.1)**3.3
    return A*E**(-3.3)
def mu_pair_f(E):
    return 0.0 if E<=1.022 else 0.0045*(E-1.022)
E_coh_pts = np.array([0.01,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.5,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0])
mu_coh_pts= np.array([0.12,0.10,0.040,0.020,0.015,0.0075,0.0048,0.0025,0.0012,0.0006,0.00045,0.00038,0.00035,0.00032,0.00028,0.00020])
def mu_coh_f(E): return np.interp(E, E_coh_pts, mu_coh_pts, left=0.12, right=0.00015)
def mu_comp_f(E): return n_e*kn_total(E)

Eg = np.linspace(0.001,1.401,1401)
mu_comp_g = np.array([mu_comp_f(e) for e in Eg])
mu_coh_g  = np.array([mu_coh_f(e) for e in Eg])
mu_pho_g  = np.array([mu_photo_f(e) for e in Eg])
mu_pri_g  = np.array([mu_pair_f(e) for e in Eg])
mu_tot_g  = mu_comp_g+mu_coh_g+mu_pho_g+mu_pri_g
mu_comp = interp1d(Eg, mu_comp_g, bounds_error=False, fill_value=(mu_comp_g[0],mu_comp_g[-1]))
mu_coh  = interp1d(Eg, mu_coh_g,  bounds_error=False, fill_value=(mu_coh_g[0],mu_coh_g[-1]))
mu_pho  = interp1d(Eg, mu_pho_g,  bounds_error=False, fill_value=(mu_pho_g[0],mu_pho_g[-1]))
mu_pri  = interp1d(Eg, mu_pri_g,  bounds_error=False, fill_value=(mu_pri_g[0],mu_pri_g[-1]))
mu_tot  = interp1d(Eg, mu_tot_g,  bounds_error=False, fill_value=(mu_tot_g[0],mu_tot_g[-1]))

E_muen = np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0,3.0])
muen_pts= np.array([4.79,1.23,0.541,0.1537,0.0682,0.0409,0.0320,0.0260,0.0255,0.0275,0.0268,0.0288,0.0295,0.0328,0.0329,0.0322,0.0310,0.0303,0.0297,0.0294,0.0284,0.0266,0.0238])
mu_en_f = interp1d(E_muen, muen_pts, kind='cubic', bounds_error=False, fill_value='extrapolate')

# ---- quantile-table inverse CDF for Compton and coherent ----
nEtab = 280
Etab = np.linspace(0.001,1.4,nEtab)
nQ = 2000
rq = np.linspace(0,1,nQ)
cos_q_comp = np.zeros((nEtab,nQ)); cos_q_coh = np.zeros((nEtab,nQ))
for i,E in enumerate(Etab):
    cm = np.linspace(-1+1e-6,1-1e-6,20000)
    dens = kn_dsig_dOm(E, cm)
    cum = np.concatenate([[0],np.cumsum(dens)])
    cum = cum/cum[-1]
    cos_q_comp[i] = np.interp(rq, cum, np.concatenate([[-1],cm,[1]]))
    dens2 = ray_dsig_dOm_shape(E, cm)
    cum2 = np.concatenate([[0],np.cumsum(dens2)])
    cum2 = cum2/cum2[-1]
    cos_q_coh[i] = np.interp(rq, cum2, np.concatenate([[-1],cm,[1]]))
# slopes for linear interp in rq
def sample_from_qtable(qtab,E,r):
    idx = np.clip(np.searchsorted(Etab,E,side='right')-1,0,nEtab-1)
    pos = np.clip(np.searchsorted(rq,r,side='right')-1,0,nQ-2)
    c0 = qtab[idx,pos]; c1 = qtab[idx,pos+1]
    dr = rq[pos+1]-rq[pos]
    return c0 + (c1-c0)*(r-rq[pos])/dr

def sample_compton_cos(E,r): return sample_from_qtable(cos_q_comp,E,r)
def sample_coherent_cos(E,r): return sample_from_qtable(cos_q_coh,E,r)

# geometry
SURF_Z=0.0; PH=30.0; CYLZ0,CYLZ1=4.9,5.1; CYLR=1.0
Vcyl = np.pi*CYLR*CYLR*(CYLZ1-CYLZ0)
Phi0_area = 10.5*10.5

def seg_in_cyl_vec(p0,p1):
    d=p1-p0; dx,dy,dz=d[:,0],d[:,1],d[:,2]
    x0,y0,z0=p0[:,0],p0[:,1],p0[:,2]
    a=dx*dx+dy*dy; b=2*(x0*dx+y0*dy); c=x0*x0+y0*y0-CYLR*CYLR
    t0=np.zeros(len(d)); t1=np.ones(len(d))
    mask = (a>1e-14)&(b*b-4*a*c>0)
    sq=np.sqrt(np.maximum(b*b-4*a*c,0)[mask])
    ta=(-b[mask]-sq)/(2*a[mask]); tb=(-b[mask]+sq)/(2*a[mask])
    t0[mask]=np.maximum(t0[mask],np.minimum(ta,tb)); t1[mask]=np.minimum(t1[mask],np.maximum(ta,tb))
    outside = (~(a>1e-14)) & (c>0); t1[outside]=0
    az=np.abs(dz)>1e-12
    dzz=np.where(az,dz,1.0)
    tza=np.where(az,(CYLZ0-z0)/dzz,0); tzb=np.where(az,(CYLZ1-z0)/dzz,0)
    t0=np.maximum(t0,np.minimum(tza,tzb)); t1=np.minimum(t1,np.maximum(tza,tzb))
    zin = az | ((z0>=CYLZ0)&(z0<=CYLZ1)); t1[~zin]=0
    return np.maximum(t1-t0,0)*np.sqrt(dx*dx+dy*dy+dz*dz)

def run_mc(N, seed=1, do_print=False):
    rng = np.random.default_rng(seed)
    t0=time.time()
    flu_pri_ph=np.zeros(nph); flu_pri_ct=np.zeros(nct)
    flu_sct_ph=np.zeros(nph); flu_sct_ct=np.zeros(nct)
    n_int={'comp':0,'photo':0,'pair':0,'ray':0}
    n_int_pri={'comp':0,'photo':0,'pair':0,'ray':0}
    n_int_sct={'comp':0,'photo':0,'pair':0,'ray':0}
    n_scat_inter=0
    E_tr_by_proc={'comp':0.0,'photo':0.0,'pair':0.0,'ray':0.0}
    E_tr_by_proc_pri={'comp':0.0,'photo':0.0,'pair':0.0,'ray':0.0}
    E_tr_by_proc_sct={'comp':0.0,'photo':0.0,'pair':0.0,'ray':0.0}
    E_tr_scat=0.0
    el_ini_all=np.zeros(nel); el_ini_pri=np.zeros(nel); el_ini_pos=np.zeros(nel)
    el_dir=np.zeros(nct)
    n_el_pri=0
    N_int_total=0
    N_int_pri_total=0; N_int_sct_total=0

    x=rng.uniform(-5,5,N); y=rng.uniform(-5,5,N); z=np.full(N,-100.0)
    dx=x; dy=y; dz=100.0; norm=np.sqrt(dx*dx+dy*dy+dz*dz)
    pos=np.stack([x,y,z],axis=1); dire=np.stack([dx/norm,dy/norm,dz/norm],axis=1)
    E=np.where(rng.random(N)<0.5,1.1732,1.3325)
    alive=np.ones(N,bool); primary=np.ones(N,bool)
    Ecut=0.002
    it=0
    while alive.any():
        it+=1
        n_act=int(alive.sum())
        Em=np.maximum(E[alive],Ecut)
        mu=mu_tot(Em)
        s=-np.log(np.maximum(rng.random(n_act),1e-30))/mu
        p1=pos[alive]+dire[alive]*s[:,None]
        L=seg_in_cyl_vec(pos[alive],p1)
        m=L>0
        if m.any():
            Em_sub=Em[m]; Lm=L[m]
            pri=primary[alive][m]
            ph_idx=np.clip(np.searchsorted(ph_edges,Em_sub,side='right')-1,0,nph-1)
            ct_idx=np.clip(np.searchsorted(ct_edges,dire[alive][m][:,2],side='right')-1,0,nct-1)
            np.add.at(flu_pri_ph,ph_idx[pri],Lm[pri]); np.add.at(flu_sct_ph,ph_idx[~pri],Lm[~pri])
            np.add.at(flu_pri_ct,ct_idx[pri],Lm[pri]); np.add.at(flu_sct_ct,ct_idx[~pri],Lm[~pri])
        outside=(p1[:,2]<SURF_Z)|(p1[:,2]>PH)|(np.abs(p1[:,0])>PH/2)|(np.abs(p1[:,1])>PH/2)
        pos=np.where(alive[:,None],p1,pos)
        do_int=alive & ~outside
        N_int_total+=int(do_int.sum())
        N_int_pri_total+=int((do_int&primary).sum()); N_int_sct_total+=int((do_int&~primary).sum())
        # process sampling over alive photons
        comp_p=mu_comp(Em)/mu_tot(Em); coh_p=mu_coh(Em)/mu_tot(Em); pho_p=mu_pho(Em)/mu_tot(Em)
        rr=rng.random(n_act)
        comp_a = rr < comp_p
        coh_a  = (~comp_a)&(rr < comp_p+coh_p)
        pho_a  = (~comp_a)&(~coh_a)&(rr < comp_p+coh_p+pho_p)
        pri_a  = (~comp_a)&(~coh_a)&(~pho_a)
        act_idx=np.where(alive)[0]
        incyl_a = (L>0)   # L computed for alive photons; in cylinder means segment inside (interaction point within cylinder? we used L>0 for segment within)
        # Interaction point is p1; to score interactions in cylinder, require p1 inside cylinder
        p1x,p1y,p1z=p1[:,0],p1[:,1],p1[:,2]
        incyl_pt = (p1x*p1x+p1y*p1y <= CYLR*CYLR) & (p1z>=CYLZ0)&(p1z<=CYLZ1)
        incyl_pt = incyl_pt & do_int[alive]
        # --- Compton ---
        if comp_a.any():
            Ecomp=Em[comp_a]
            costh=sample_compton_cos(Ecomp, rng.random(int(comp_a.sum())))
            Ep=Ecomp/(1+(Ecomp/mec2)*(1-costh))
            u_old=dire[alive][comp_a]
            cross=np.cross(u_old,np.array([0.,0.,1.])); nc=np.linalg.norm(cross,axis=1)
            fall=nc<1e-8; e1=np.zeros_like(cross)
            e1[~fall]=cross[~fall]/nc[~fall,None]; e1[fall]=np.array([1.,0.,0.])
            e2=np.cross(u_old,e1); phi=rng.uniform(0,TWOPI,int(comp_a.sum()))
            sinth=np.sqrt(np.maximum(1-costh*costh,0))
            u_new=u_old*costh[:,None]+(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)*sinth[:,None]
            p_e=Ecomp[:,None]*u_old-Ep[:,None]*u_new
            T_e=Ecomp-Ep
            pn=np.linalg.norm(p_e,axis=1)
            el_dir_rec=p_e/np.maximum(pn[:,None],1e-30)
            set_idx=act_idx[comp_a]
            E[set_idx]=Ep; dire[set_idx]=u_new; primary[set_idx]=False
            # record cylinder interactions
            ic = incyl_pt[comp_a]
            if ic.any():
                Tsub=T_e[ic]; sct_parent = ~primary[act_idx[comp_a]][ic]  # primary already set False; use saved
                # capture parent before change: we changed primary above; save pri_comp
            # (handled below with saved array)
        # do records separately using saved prime status: recompute from n (primary before update saved)
        # Simpler: recompute parent flags before modifications in a list
    # end (placeholder)
    return None
print("setup ok")
