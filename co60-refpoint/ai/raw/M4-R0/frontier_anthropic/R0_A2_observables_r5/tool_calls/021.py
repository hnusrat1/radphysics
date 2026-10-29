
final = r'''
import numpy as np, pickle
from mcana import *
from mcmod import *
import ana as A

def combine(tal, ptal, nh):
    C={}
    s = {k:(v/nh if isinstance(v,(float,np.ndarray,int,np.float64)) else v) for k,v in tal.items() if k not in ('nh','ncoll','nseg')}
    p = ptal
    def g(key):
        return np.asarray(s[key]) + np.asarray(p[key])
    C['q_fine_em']=g('q_fine_em'); C['q_fine_all']=g('q_fine_all')
    C['q_fine_em_cos']=g('q_fine_em_cos'); C['q_fine_all_cos']=g('q_fine_all_cos')
    C['el_T_all']=g('el_T_all'); C['el_C_all']=g('el_C_all')
    C['el_w_all']=s['el_w_all']+p['el_w_all']; C['el_wT_all']=s['el_wT_all']+p['el_wT_all']
    C['el_T_prim']=p['el_T_all']; C['el_w_prim']=p['el_w_all']; C['el_wT_prim']=p['el_wT_all']
    return C, s, p

def main(tal, ptal, Pper_src, nh):
    C, s, p = combine(tal, ptal, nh)
    out={}
    Pphi0 = A.primary_fluence_per_phi0()          # per Phi0, per line
    Psrc = Pphi0*Phi0_src                          # per source photon per line
    # ---------- L1 ----------
    scat_per_phi0 = (tal['arr_tot']/nh)/Phi0_src
    prim_per_phi0 = Pphi0.sum()
    total = prim_per_phi0+scat_per_phi0
    Ep = np.sum(Pphi0*LINES); Es=(tal['arr_E']/nh)/Phi0_src
    out['L1_primary_fluence_per_phi0']=prim_per_phi0
    out['L1_total_fluence_per_phi0']=total
    out['L1_scattered_fraction_fluence']=scat_per_phi0/total
    out['L1_scattered_fraction_energy_fluence']=Es/(Es+Ep)
    out['L1_mean_photon_energy_MeV']=(Es+Ep)/total
    prim_spec=np.zeros(28); prim_spec[23]=Pphi0[0]; prim_spec[26]=Pphi0[1]
    out['L1_photon_fluence_spectrum_primary']=prim_spec.tolist()
    out['L1_photon_fluence_spectrum_scattered']=((tal['arr_spec']/nh)/Phi0_src).tolist()
    ac=np.zeros(10); ac[9]=1.0
    out['L1_angular_primary']=ac.tolist()
    sc=(tal['arr_cos']/nh); out['L1_angular_scattered']=(sc/sc.sum()).tolist()
    # ---------- L2 ----------
    Ei=LINES
    prim_cnt = dict(compton=np.sum(Psrc*mu_inc(Ei)), photoelectric=np.sum(Psrc*mu_pe(Ei)),
                    pair=np.sum(Psrc*mu_pair(Ei)), rayleigh=np.sum(Psrc*mu_coh(Ei)))
    sc_cnt = dict(compton=tal['cnt_c']/nh, photoelectric=tal['cnt_pe']/nh, pair=tal['cnt_pair']/nh, rayleigh=tal['cnt_r']/nh)
    prim_etr = dict(compton=np.sum(Psrc*mu_tr_c(Ei)), photoelectric=np.sum(Psrc*mu_tr_pe(Ei)),
                    pair=np.sum(Psrc*mu_tr_pair(Ei)), rayleigh=0.0)
    sc_etr = dict(compton=tal['etr_c']/nh, photoelectric=tal['etr_pe']/nh, pair=tal['etr_pair']/nh, rayleigh=0.0)
    def frac(d):
        t=sum(d.values()); return {k:v/t for k,v in d.items()}
    all_cnt={k:prim_cnt[k]+sc_cnt[k] for k in prim_cnt}
    all_etr={k:prim_etr[k]+sc_etr[k] for k in prim_cnt}
    out['L2_interaction_fractions']=frac(all_cnt)
    out['L2_energy_transfer_fractions']=frac(all_etr)
    out['L2_interaction_fractions_primary']=frac(prim_cnt)
    out['L2_interaction_fractions_scattered']=frac(sc_cnt)
    out['L2_scattered_share_of_interactions']=sum(sc_cnt.values())/sum(all_cnt.values())
    # ---------- L3 ----------
    h=C['el_T_all']; out['L3_initial_spectrum']=(h/h.sum()).tolist()
    hp=C['el_T_prim']; out['L3_initial_spectrum_parent_primary']=(hp/hp.sum()).tolist()
    out['L3_mean_initial_energy_MeV']=C['el_wT_all']/C['el_w_all']
    out['L3_mean_initial_energy_parent_primary_MeV']=C['el_wT_prim']/C['el_w_prim']
    hc=C['el_C_all']; out['L3_initial_direction']=(hc/hc.sum()).tolist()
    # ---------- L4 ----------
    dF = FINE[1:]-FINE[:-1]; cF=np.sqrt(FINE[:-1]*FINE[1:])
    qe = C['q_fine_em']/dF
    ER = 0.01; Emax=1.4
    idx=np.where((cF>=0.0099)&(cF<=Emax))[0]
    Psi=np.zeros(cF.size); Psid=np.zeros(cF.size)
    Nab=0.0; Nabd=0.0
    ne_=ne
    for i in idx[::-1]:
        Ti=cF[i]
        js=idx[(cF[idx]>=2*Ti)]
        Qd=0.0
        if js.size:
            Qd = ne_*np.sum(Psi[js]*moller(cF[js],Ti)*dF[js])
        Qp=qe[i]
        Qt=Qp+Qd
        S=S_tot(Ti)
        Psi[i]=(Nab+0.5*Qt*dF[i])/S
        Psid[i]=(Nabd+0.5*Qd*dF[i])/S
        Nab+=Qt*dF[i]; Nabd+=Qd*dF[i]
    Psi_src = Psi
    L4tot = np.sum(Psi*dF)/Phi0_src
    out['L4_electron_fluence_per_phi0']=L4tot
    spec=np.zeros(25)
    for b in range(25):
        m=(cF>=EL_EDGES[b])&(cF<EL_EDGES[b+1])
        spec[b]=np.sum(Psi[m]*dF[m])/Phi0_src
    out['L4_electron_fluence_spectrum']=spec.tolist()
    out['L4_delta_fraction']=np.sum(Psid*dF)/np.sum(Psi*dF)
    out['L4_mean_electron_energy_MeV']=np.sum(Psi*dF*cF)/np.sum(Psi*dF)
    # angular: track-length (to 10 keV) weighted source directions
    qc = C['q_fine_em_cos']/dF[:,None]*0 + C['q_fine_em_cos']   # per bin weights (not density)
    trk = np.maximum(CSDA(np.maximum(cF,0.0101))-CSDA(0.01),0)
    angw = np.sum(qc*trk[:,None],axis=0)
    out['L4_angular']=(angw/angw.sum()).tolist()
    out['L4_radiative_to_collision']=np.sum(Psi*dF*S_rad(cF))/np.sum(Psi*dF*S_col(cF))
    # ---------- L5 ----------
    Kc_prim = np.sum(Psrc*Ei*mu_en_all(Ei)); Kc_sc = tal['Kcol_sc']/nh/Phi0_src
    K_prim = np.sum(Psrc*Ei*mu_tr_all(Ei)); K_sc = tal['K_sc']/nh
    # per Phi0 units
    Kcol_phi0 = (Kc_prim+tal['Kcol_sc']/nh)/Phi0_src
    K_phi0 = (K_prim+tal['K_sc']/nh)/Phi0_src
    conv=1.602176634e-10
    out['L5_Kcol_over_phi0_Gy_cm2']=conv*Kcol_phi0
    out['L5_K_over_phi0_Gy_cm2']=conv*K_phi0
    # deposited energy by electrons (track integral) as a check
    Dtrack = np.sum(Psi*dF*S_col(cF))/Phi0_src
    out['_Dtrack_over_Kcol']=Dtrack/Kcol_phi0
    out['L5_g']=1-Kcol_phi0/K_phi0
    out['L5_beta']=1.0
    out['L5_D_over_phi0_Gy_cm2']=out['L5_Kcol_over_phi0_Gy_cm2']*out['L5_beta']
    out['L5_dose_fraction_parent_scattered']=(tal['Kcol_sc']/nh)/(tal['Kcol_sc']/nh+Kc_prim)
    # Spencer-Attix water/air
    Dl=0.01
    Sw=Srestr(cF,Dl,75.0,ZA_W,DEL(cF)); Sa=Srestr(cF,Dl,85.7,ZA_AIR,np.zeros_like(cF))
    m=cF>=Dl
    iD=np.argmin(np.abs(cF-Dl))
    Tend_w=Dl*Psi[iD]*Sw[iD]; Tend_a=Dl*Psi[iD]*Sa[iD]
    sw=(np.sum(Psi[m]*dF[m]*Sw[m])+Tend_w)/(np.sum(Psi[m]*dF[m]*Sa[m])+Tend_a)
    out['L5_s_w_air']=sw
    out['_sw_no_track_end']=np.sum(Psi[m]*dF[m]*Sw[m])/np.sum(Psi[m]*dF[m]*Sa[m])
    return out, dict(Psi=Psi,cF=cF,dF=dF,Psid=Psid,Kc_prim=Kc_prim,Pphi0=Pphi0,Psrc=Psrc)
'''
open('/tmp/work/final.py','w').write(final)
import sys, time, pickle; sys.path.insert(0,'/tmp/work')
import importlib, mcana, mcmod, ana as A, final as F
importlib.reload(mcana); importlib.reload(mcmod); importlib.reload(A); importlib.reload(F)
from mcana import *
from mcmod import *
tal=pickle.load(open('/tmp/work/tal_run1.pkl','rb'))
rng=np.random.default_rng(7)
ptal,Pper=A.primary_electron_tally(rng, Ns=400000)
out,aux=F.main(tal, ptal, Pper, tal['nh'])
for k,v in out.items():
    if isinstance(v,list): print(k, np.round(np.array(v),5).tolist()[:28])
    elif isinstance(v,dict): print(k, {kk:round(vv,5) for kk,vv in v.items()})
    else: print(k, v)
