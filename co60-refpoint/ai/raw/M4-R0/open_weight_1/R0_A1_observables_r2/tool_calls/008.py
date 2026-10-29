
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
def mu_coh_f(E): return float(np.interp(E, E_coh_pts, mu_coh_pts, left=0.12, right=0.00015))
def mu_comp_f(E): return n_e*kn_total(E)

Eg = np.linspace(0.001,1.401,1401)
mu_comp_g = np.array([mu_comp_f(e) for e in Eg]); mu_coh_g = np.array([mu_coh_f(e) for e in Eg])
mu_pho_g = np.array([mu_photo_f(e) for e in Eg]); mu_pri_g = np.array([mu_pair_f(e) for e in Eg])
mu_tot_g = mu_comp_g+mu_coh_g+mu_pho_g+mu_pri_g
mu_comp = interp1d(Eg, mu_comp_g, bounds_error=False, fill_value=(mu_comp_g[0],mu_comp_g[-1]))
mu_coh  = interp1d(Eg, mu_coh_g,  bounds_error=False, fill_value=(mu_coh_g[0],mu_coh_g[-1]))
mu_pho  = interp1d(Eg, mu_pho_g,  bounds_error=False, fill_value=(mu_pho_g[0],mu_pho_g[-1]))
mu_pri  = interp1d(Eg, mu_pri_g,  bounds_error=False, fill_value=(mu_pri_g[0],mu_pri_g[-1]))
mu_tot  = interp1d(Eg, mu_tot_g,  bounds_error=False, fill_value=(mu_tot_g[0],mu_tot_g[-1]))

E_muen = np.array([0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0,3.0])
muen_pts= np.array([4.79,1.23,0.541,0.1537,0.0682,0.0409,0.0320,0.0260,0.0255,0.0275,0.0268,0.0288,0.0295,0.0328,0.0329,0.0322,0.0310,0.0303,0.0297,0.0294,0.0284,0.0266,0.0238])
mu_en_f = interp1d(E_muen, muen_pts, kind='cubic', bounds_error=False, fill_value='extrapolate')

# ---- quantile tables ----
nEtab = 280; Etab = np.linspace(0.001,1.4,nEtab)
nQ = 2000; rq = np.linspace(0,1,nQ)
cos_q_comp = np.zeros((nEtab,nQ)); cos_q_coh = np.zeros((nEtab,nQ))
cm = np.linspace(-1+1e-6,1-1e-6,20000)
for i,E in enumerate(Etab):
    dens = kn_dsig_dOm(E, cm); cum = np.concatenate([[0],np.cumsum(dens)]); cum=cum/cum[-1]
    xp = np.concatenate([[-1.0],cm])
    cos_q_comp[i] = np.interp(rq, cum, xp)
    dens2 = ray_dsig_dOm_shape(E, cm); cum2 = np.concatenate([[0],np.cumsum(dens2)]); cum2=cum2/cum2[-1]
    cos_q_coh[i] = np.interp(rq, cum2, xp)
def sample_from_qtable(qtab,E,r):
    idx = np.clip(np.searchsorted(Etab,E,side='right')-1,0,nEtab-1)
    pos = np.clip(np.searchsorted(rq,r,side='right')-1,0,nQ-2)
    c0 = qtab[idx,pos]; c1 = qtab[idx,pos+1]
    dr = rq[pos+1]-rq[pos]
    return c0 + (c1-c0)*(r-rq[pos])/dr
def sample_compton_cos(E,r): return sample_from_qtable(cos_q_comp,E,r)
def sample_coherent_cos(E,r): return sample_from_qtable(cos_q_coh,E,r)

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

def run_mc(N, seed=1):
    rng = np.random.default_rng(seed)
    t0=time.time()
    flu_pri_ph=np.zeros(nph); flu_pri_ct=np.zeros(nct)
    flu_sct_ph=np.zeros(nph); flu_sct_ct=np.zeros(nct)
    n_int={'comp':0,'photo':0,'pair':0,'ray':0}
    n_int_pri={'comp':0,'photo':0,'pair':0,'ray':0}
    n_int_sct={'comp':0,'photo':0,'pair':0,'ray':0}
    n_scat_inter=0; N_int_cyl=0
    E_tr_by_proc={'comp':0.0,'photo':0.0,'pair':0.0,'ray':0.0}
    E_tr_by_proc_pri={'comp':0.0,'photo':0.0,'pair':0.0,'ray':0.0}
    E_tr_by_proc_sct={'comp':0.0,'photo':0.0,'pair':0.0,'ray':0.0}
    E_tr_scat=0.0
    el_ini_all=np.zeros(nel); el_ini_pri=np.zeros(nel); el_ini_pos=np.zeros(nel)
    el_dir=np.zeros(nct)
    n_el_pri=0
    N_el_e=0; N_el_pos=0

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
        p1x,p1y,p1z=p1[:,0],p1[:,1],p1[:,2]
        incyl_pt = (p1x*p1x+p1y*p1y <= CYLR*CYLR) & (p1z>=CYLZ0)&(p1z<=CYLZ1) & do_int[alive]
        N_int_cyl += int(incyl_pt.sum())
        comp_p=mu_comp(Em)/mu_tot(Em); coh_p=mu_coh(Em)/mu_tot(Em); pho_p=mu_pho(Em)/mu_tot(Em)
        rr=rng.random(n_act)
        comp_a = rr < comp_p
        coh_a  = (~comp_a)&(rr < comp_p+coh_p)
        pho_a  = (~comp_a)&(~coh_a)&(rr < comp_p+coh_p+pho_p)
        pri_a  = (~comp_a)&(~coh_a)&(~pho_a)
        act_idx=np.where(alive)[0]
        pri_before = primary[alive].copy()
        # ---------- Compton ----------
        if comp_a.any():
            Ecomp=Em[comp_a]; pri_comp=pri_before[comp_a]
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
            ic=incyl_pt[comp_a]
            if ic.any():
                Tsub=T_e[ic]; n_el_e += int(ic.sum())
                n_int['comp']+=int(ic.sum())
                elbin=np.clip(np.searchsorted(el_edges,Tsub,side='right')-1,0,nel-1)
                np.add.at(el_ini_all,elbin,1.0)
                ctbin=np.clip(np.searchsorted(ct_edges,el_dir_rec[ic][:,2],side='right')-1,0,nct-1)
                np.add.at(el_dir,ctbin,1.0)
                E_tr_by_proc['comp']+=Tsub.sum()
                pc=pri_comp[ic]
                n_int_pri['comp']+=int(pc.sum()); n_int_sct['comp']+=int((~pc).sum())
                E_tr_by_proc_pri['comp']+=Tsub[pc].sum(); E_tr_by_proc_sct['comp']+=Tsub[~pc].sum()
                if pc.any():
                    elb=np.clip(np.searchsorted(el_edges,Tsub[pc],side='right')-1,0,nel-1)
                    np.add.at(el_ini_pri,elb,1.0); n_el_pri+=int(pc.sum())
                E_tr_scat += Tsub[~pc].sum()
        # ---------- Coherent ----------
        if coh_a.any():
            Eco=Em[coh_a]; pri_coh=pri_before[coh_a]
            costh=sample_coherent_cos(Eco, rng.random(int(coh_a.sum())))
            u_old=dire[alive][coh_a]
            cross=np.cross(u_old,np.array([0.,0.,1.])); nc=np.linalg.norm(cross,axis=1)
            fall=nc<1e-8; e1=np.zeros_like(cross)
            e1[~fall]=cross[~fall]/nc[~fall,None]; e1[fall]=np.array([1.,0.,0.])
            e2=np.cross(u_old,e1); phi=rng.uniform(0,TWOPI,int(coh_a.sum()))
            sinth=np.sqrt(np.maximum(1-costh*costh,0))
            u_new=u_old*costh[:,None]+(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)*sinth[:,None]
            set_idx=act_idx[coh_a]
            dire[set_idx]=u_new; primary[set_idx]=False
            ic=incyl_pt[coh_a]
            if ic.any():
                n_int['ray']+=int(ic.sum()); n_scat_inter? 
            # above syntax error placeholder
        # ---------- Photoelectric ----------
        if pho_a.any():
            Ephoto=Em[pho_a]; pri_pho=pri_before[pho_a]
            # isotropic direction
            u_iso = rng.standard_normal((int(pho_a.sum()),3)); u_iso/=np.linalg.norm(u_iso,axis=1)[:,None]
            Tsub=Ephoto-0.000532
            ic=incyl_pt[pho_a]
            if ic.any():
                n_int['photo']+=int(ic.sum()); n_el_e+=int(ic.sum())
                elbin=np.clip(np.searchsorted(el_edges,Tsub[ic],side='right')-1,0,nel-1)
                np.add.at(el_ini_all,elbin,1.0)
                ctbin=np.clip(np.searchsorted(ct_edges,u_iso[ic][:,2],side='right')-1,0,nct-1)
                np.add.at(el_dir,ctbin,1.0)
                E_tr_by_proc['photo']+=Tsub[ic].sum()
                pc=pri_pho[ic]
                n_int_pri['photo']+=int(pc.sum()); n_int_sct['photo']+=int((~pc).sum())
                E_tr_by_proc_pri['photo']+=Tsub[ic][pc].sum(); E_tr_by_proc_sct['photo']+=Tsub[ic][~pc].sum()
                if pc.any():
                    elb=np.clip(np.searchsorted(el_edges,Tsub[ic][pc],side='right')-1,0,nel-1)
                    np.add.at(el_ini_pri,elb,1.0); n_el_pri+=int(pc.sum())
                E_tr_scat+=Tsub[ic][~pc].sum()
            set_idx=act_idx[pho_a]
            alive[set_idx]=False
        # ---------- Pair ----------
        if pri_a.any():
            Epair=Em[pri_a]; pr_ex=Epair-1.022
            np_=int(pri_a.sum())
            share=rng.random(np_)*pr_ex   # electron kinetic
            Te_=share; Tp_=pr_ex-share
            # directions forward-ish
            sig=0.15
            cthe=np.cos(sig*rng.standard_normal(np_)); ctpos=np.cos(sig*rng.standard_normal(np_))
            ic=incyl_pt[pri_a]
            if ic.any():
                n_int['pair']+=int(ic.sum())
                n_el_e+=int(ic.sum()); n_el_pos+=int(ic.sum())
                elbin=np.clip(np.searchsorted(el_edges,Te_[ic],side='right')-1,0,nel-1)
                np.add.at(el_ini_all,elbin,1.0)
                elbinp=np.clip(np.searchsorted(el_edges,Tp_[ic],side='right')-1,0,nel-1)
                np.add.at(el_ini_pos,elbinp,1.0)
                cb_e=np.clip(np.searchsorted(ct_edges,cthe[ic],side='right')-1,0,nct-1)
                cb_p=np.clip(np.searchsorted(ct_edges,ctpos[ic],side='right')-1,0,nct-1)
                np.add.at(el_dir,cb_e,1.0); np.add.at(el_dir,cb_p,1.0)
                tr=(Epair-1.022)[ic]
                E_tr_by_proc['pair']+=tr.sum()
                # pair from primary photon: parent primary (pair only from high E; all from primary or scattered possible)
                pc=pri_before[pri_a][ic]
                n_int_pri['pair']+=int(pc.sum()); n_int_sct['pair']+=int((~pc).sum())
                E_tr_by_proc_pri['pair']+=tr[pc].sum(); E_tr_by_proc_sct['pair']+=tr[~pc].sum()
                if pc.any():
                    # record electron only to el_ini_pri for parent-primary? Layer3 parent_primary includes e+ too; include both
                    elb=np.clip(np.searchsorted(el_edges,Te_[ic][pc],side='right')-1,0,nel-1)
                    np.add.at(el_ini_pri,elb,1.0)
                    elbp=np.clip(np.searchsorted(el_edges,Tp_[ic][pc],side='right')-1,0,nel-1)
                    np.add.at(el_ini_pri,elbp,1.0)
                    n_el_pri+=int(2*pc.sum())
                E_tr_scat+=tr[~pc].sum()
            set_idx=act_idx[pri_a]
            alive[set_idx]=False
        # photons that left cube: dead
        alive[outside]=False
        lowE=alive&(E<Ecut); alive[lowE]=False
        if do_print and it%500==0:
            print('iter',it,'alive',int(alive.sum()),'time',time.time()-t0)
    # scattered interaction count: N_int_cyl minus primary interactions in cylinder
    tot_pri_cyl = sum(n_int_pri.values()); tot_sct_cyl = sum(n_int_sct.values())
    n_scat_inter = tot_sct_cyl
    return dict(flu_pri_ph=flu_pri_ph, flu_sct_ph=flu_sct_ph, flu_pri_ct=flu_pri_ct, flu_sct_ct=flu_sct_ct,
                n_int=n_int, n_int_pri=n_int_pri, n_int_sct=n_int_sct, n_scat_inter=n_scat_inter, N_int_cyl=N_int_cyl,
                E_tr_by_proc=E_tr_by_proc, E_tr_by_proc_pri=E_tr_by_proc_pri, E_tr_by_proc_sct=E_tr_by_proc_sct,
                E_tr_scat=E_tr_scat,
                el_ini_all=el_ini_all, el_ini_pri=el_ini_pri, el_ini_pos=el_ini_pos, el_dir=el_dir,
                n_el_pri=n_el_pri, n_el_e=n_el_e, n_el_pos=n_el_pos,
                N=N, t=time.time()-t0)

res = run_mc(50000, seed=11)
print('time', round(res['t'],1))
print('N',res['N'])
print('flu pri total (cm)', res['flu_pri_ph'].sum(), 'flu sct total', res['flu_sct_ph'].sum())
print('interactions', res['n_int'], res['n_int_pri'], res['n_int_sct'])
print('N_int_cyl', res['N_int_cyl'], 'scat share', res['n_scat_inter']/max(res['N_int_cyl'],1))
print('E_tr by proc', res['E_tr_by_proc'])
print('n_el_e', res['n_el_e'], 'n_el_pos', res['n_el_pos'])
print('time per 10k', res['t']/5)
