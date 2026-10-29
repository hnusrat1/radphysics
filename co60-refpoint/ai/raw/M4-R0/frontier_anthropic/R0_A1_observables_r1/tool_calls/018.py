
anal = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import phys, mc, elec
from phys import re, mc2, ne, Lines, mu_c, mu_pe, mu_pair, mu_coh, mu_tot, ETR, OMEGA, mu_tr_c
from elec import f_moller, beta2, S_col_BS, S_rad_w, S_res, ne_air
PE=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4]
EE=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4]
CE=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
def analyse(N, seed, Nel=200000):
    rng=np.random.default_rng(seed)
    res=mc.run(N,[0.5,5.0],rng,store_for=5.0)
    A=res['arr']
    Fn=OMEGA*105.0**2/N          # per-emitted-photon -> per Phi0
    Eа=None
    E=A['E']; u=A['u']; W=A['W']*Fn; typ=A['typ']
    mu1,mu2=mu_tot(Lines)
    # primary (per Phi0)
    Pw=0.5*np.exp(-mu_tot(Lines)*5.0)
    out={}
    # ---- Layer 1
    prim_tot=Pw.sum()
    scat_tot=W.sum()
    out['L1_primary_fluence_per_phi0']=prim_tot
    out['L1_total_fluence_per_phi0']=prim_tot+scat_tot
    out['L1_scattered_fraction_fluence']=scat_tot/(prim_tot+scat_tot)
    eprim=np.sum(Pw*Lines); escat=np.sum(W*E)
    out['L1_scattered_fraction_energy_fluence']=escat/(eprim+escat)
    out['L1_mean_photon_energy_MeV']=(eprim+escat)/(prim_tot+scat_tot)
    sp=np.zeros(28); idx=np.searchsorted(PE,Lines,side='right')-1
    for i,p in zip(idx,Pw): sp[i]+=p
    out['L1_photon_fluence_spectrum_primary']=sp
    ss,_=np.histogram(E,bins=PE,weights=W); out['L1_photon_fluence_spectrum_scattered']=ss
    out['L1_angular_primary']=[0]*9+[1.0]
    cz=u[:,2]
    ca,_=np.histogram(cz,bins=CE,weights=W); out['L1_angular_scattered']=ca/ca.sum()
    # ---- Layer 2
    def fracs(Ev,w,kind):
        if kind=='c': return np.sum(w*mu_c(Ev))
        if kind=='p': return np.sum(w*mu_pe(Ev))
        if kind=='pair': return np.sum(w*mu_pair(Ev))
        if kind=='r': return np.sum(w*mu_coh(Ev))
    def interact(Ev,w):
        return {k:fracs(Ev,w,k) for k in ['c','p','pair','r']}
    def energy_tr(Ev,w):
        return {'c':np.sum(w*ETR(Ev)),'p':np.sum(w*mu_pe(Ev)*Ev),'pair':np.sum(w*mu_pair(Ev)*(Ev-1.022)),'r':0.0}
    Ip=interact(Lines,Pw); Is=interact(E,W)
    tot={k:Ip[k]+Is[k] for k in Ip}
    def norm(d):
        s=sum(d.values()); return {k:d[k]/s for k in d}
    keymap={'c':'compton','p':'photoelectric','pair':'pair','r':'rayleigh'}
    out['L2_interaction_fractions']={keymap[k]:v for k,v in norm(tot).items()}
    Tp=energy_tr(Lines,Pw); Ts=energy_tr(E,W)
    Tt={k:Tp[k]+Ts[k] for k in Tp}
    out['L2_energy_transfer_fractions']={keymap[k]:v for k,v in norm(Tt).items()}
    out['L2_interaction_fractions_primary']={keymap[k]:v for k,v in norm(Ip).items()}
    out['L2_interaction_fractions_scattered']={keymap[k]:v for k,v in norm(Is).items()}
    out['L2_scattered_share_of_interactions']=sum(Is.values())/sum(tot.values())
    # ---- electrons produced at P: arrivals (all types) + primary lines
    def sample_compton(Ev,uv,wv,rng):
        c=kn_cos(Ev,rng); phi=rng.uniform(0,2*np.pi,Ev.size)
        n=rotate(uv,c,phi)
        Eo=Ev/(1+Ev/mc2*(1-c))
        pe=Ev[:,None]*uv-Eo[:,None]*n
        T=Ev-Eo
        dirn=pe/np.linalg.norm(pe,axis=1)[:,None]
        return T,dirn,wv
    from mc import kn_sample_cos as kn_cos, rotate
    # scattered arrivals
    Tlist=[];Wlist=[];Dlist=[];Plist=[]  # parent flags: 1 = primary, 0 = scattered
    for kind in ['c','p','pair']:
        pass
    # Compton from arrivals
    wc=W*mu_c(E)
    T1,D1,w1=sample_compton(E,u,wc,rng)
    Tlist.append(T1);Dlist.append(D1);Wlist.append(w1);Plist.append(np.zeros(T1.size,int))
    # photoelectric from arrivals
    wp=W*mu_pe(E)
    Tlist.append(E.copy());Dlist.append(u.copy());Wlist.append(wp);Plist.append(np.zeros(E.size,int))
    # pair from arrivals: e- and e+ share Ekin=E-1.022, uniform split
    wpa=W*mu_pair(E); Ek=np.clip(E-1.022,0,None); fr=rng.uniform(size=E.size)
    Tlist.append(Ek*fr);Dlist.append(u.copy());Wlist.append(wpa);Plist.append(np.zeros(E.size,int)) # e-
    Tlist.append(Ek*(1-fr));Dlist.append(u.copy());Wlist.append(wpa.copy());Plist.append(np.zeros(E.size,int)) # e+
    ise_m=np.zeros(0)
    ispos=[np.zeros(E.size,bool)]
    # primary lines (sample Compton and pair/photo analytically via sampling)
    for li,(El,w0) in enumerate(zip(Lines,Pw)):
        nS=Nel
        Ev=np.full(nS,El); uv=np.tile([0,0,1.0],(nS,1))
        T2,D2,_=sample_compton(Ev,uv,None,rng)
        wsc=np.full(nS, w0*mu_c(El)/nS)
        Tlist.append(T2);Dlist.append(D2);Wlist.append(wsc);Plist.append(np.ones(nS,int))
        Tlist.append(np.array([El]));Dlist.append(np.array([[0,0,1.0]]));Wlist.append(np.array([w0*mu_pe(El)]));Plist.append(np.ones(1,int))
        if El>1.022:
            Ek2=El-1.022
            Tlist.append(np.array([Ek2*0.5]));Dlist.append(np.array([[0,0,1.0]]));Wlist.append(np.array([w0*mu_pair(El)]));Plist.append(np.ones(1,int))
            Tlist.append(np.array([Ek2*0.5]));Dlist.append(np.array([[0,0,1.0]]));Wlist.append(np.array([w0*mu_pair(El)]));Plist.append(np.ones(1,int))
    # positron flags: pair e+ were appended as 2nd sets; we track separately below
    return dict(out=out, Tlist=Tlist, Dlist=Dlist, Wlist=Wlist, Plist=Plist, E=E, W=W, u=u, typ=typ, Pw=Pw, Fn=Fn, N=N, res=res)
'''
open('/tmp/work/anal.py','w').write(anal)
print("written")
