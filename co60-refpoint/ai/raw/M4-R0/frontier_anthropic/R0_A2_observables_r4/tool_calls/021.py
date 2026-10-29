
ana2 = r'''
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, pickle
from phys import *
from tables import Y_T, R_T
import tables as TB
from mc3 import EDGES_PH, EDGES_EL, EDGES_CT, sample_KN, rotate, DEPTHS, VOL
from scipy.integrate import quad
_tab={k:v for k,v in np.load('/tmp/work/tables.npz').items()}
_lE=np.log(_tab['E'])
def ftr_f(E): return np.interp(np.log(E),_lE,_tab['ftr'])
def kcf_f(E): return np.interp(np.log(E),_lE,_tab['kcf'])
SM=pickle.load(open('/tmp/work/SM.pkl','rb'))
Om=0.009975072683099257
F=Om*105**2/SM['N']
rng=np.random.default_rng(7)
LN=np.array([1.1732,1.3325])
mut=mu_tot(LN)
phi=0.5*np.exp(-mut*5.0)          # primary per Phi0 at 5 cm, per line
# ---------- L1
prim_tot=phi.sum()
scat_hE=SM['hE'][4]*F
scat_tot=SM['g'][4]*F
scat_E=SM['gE'][4]*F
prim_E=(phi*LN).sum()
total=prim_tot+scat_tot
L1={}
L1['L1_primary_fluence_per_phi0']=prim_tot
L1['L1_total_fluence_per_phi0']=total
L1['L1_scattered_fraction_fluence']=scat_tot/total
L1['L1_scattered_fraction_energy_fluence']=scat_E/(scat_E+prim_E)
L1['L1_mean_photon_energy_MeV']=(scat_E+prim_E)/total
sp=np.zeros(28); sp[23]=phi[0]; sp[26]=phi[1]
L1['L1_photon_fluence_spectrum_primary']=sp
L1['L1_photon_fluence_spectrum_scattered']=scat_hE
ap=np.zeros(10); ap[9]=1.0
L1['L1_angular_primary']=ap
L1['L1_angular_scattered']=SM['hC'][4]/SM['hC'][4].sum()
# ---------- L2
# primary interactions per line
pr_int=np.array([[phi[i]*mu_inc(LN[i]), phi[i]*mu_pe(LN[i]), phi[i]*mu_pair(LN[i]), phi[i]*mu_coh(LN[i])] for i in range(2)]).sum(0)
pr_trE=np.array([phi[i]*(mu_inc(LN[i])*LN[i]*ftr_f(LN[i])), phi[i]*mu_pe(LN[i])*LN[i], phi[i]*mu_pair(LN[i])*max(LN[i]-1.022,0), 0.0] for i in range(2)).sum(0)
sc_int=SM['int'][4]*F; sc_trE=SM['trE'][4]*F
def fr(v): return {'compton':v[0]/v.sum(),'photoelectric':v[1]/v.sum(),'pair':v[2]/v.sum(),'rayleigh':v[3]/v.sum()}
L2={}
L2['L2_interaction_fractions']=fr(pr_int+sc_int)
L2['L2_energy_transfer_fractions']=fr(pr_trE+sc_trE)
L2['L2_interaction_fractions_primary']=fr(pr_int)
L2['L2_interaction_fractions_scattered']=fr(sc_int)
L2['L2_scattered_share_of_interactions']=sc_int.sum()/(pr_int.sum()+sc_int.sum())
print('pr_int',pr_int,'sc_int',sc_int)
print('pr_trE',pr_trE,'sc_trE',sc_trE)
# ---------- electrons: primary samples (for L3 and L4 angular)
def elec_from(E,u,w,rng):
    """E: energies, u: directions (n,3), w: weights. returns T, cos_e"""
    eps,cs=sample_KN(E,rng)
    phi_=rng.uniform(0,2*np.pi,E.size)
    nout=rotate(u,cs,phi_)
    T=E*(1-eps)
    pe_=E[:,None]*u-(E*eps)[:,None]*nout
    cos_e=pe_[:,2]/np.linalg.norm(pe_,axis=1)
    return T,cos_e
nsamp=600000
Ep=[];Up=[];Wp=[]
for i in range(2):
    Ei=np.full(nsamp,LN[i]); ui=np.tile([0,0,1.0],(nsamp,1))
    Ep.append(Ei); Up.append(ui); Wp.append(np.full(nsamp,phi[i]*mu_inc(LN[i])/nsamp))
Ep=np.concatenate(Ep);Up=np.concatenate(Up);Wp=np.concatenate(Wp)
Tpc,cpc=elec_from(Ep,Up,Wp,rng)
# primary PE electrons
pe_w=np.array([phi[i]*mu_pe(LN[i]) for i in range(2)])
Tpe=LN; cpe=np.array([1.0,1.0])
# L3 primary histograms
hp=np.histogram(Tpc,bins=EDGES_EL,weights=Wp)[0]
hp=hp+np.histogram(Tpe,bins=EDGES_EL,weights=pe_w)[0]
wp_all=Wp.sum()+pe_w.sum()
wpT_all=(Wp*Tpc).sum()+(pe_w*Tpe).sum()
hpC=np.bincount(np.clip(np.searchsorted(EDGES_CT,cpc,side='right')-1,0,9),weights=Wp,minlength=10)
hpC+=np.bincount(np.clip(np.searchsorted(EDGES_CT,np.ones(2),side='right')-1,0,9),weights=pe_w,minlength=10)
# scattered (MC)
hs=SM['L3_hT']*F; ws=SM['L3_w']*F; wsT=SM['L3_wT']*F; hsC=SM['L3_hC']*F
L3={}
allh=hp+hs
L3['L3_initial_spectrum']=allh/allh.sum()
L3['L3_initial_spectrum_parent_primary']=hp/hp.sum()
L3['L3_mean_initial_energy_MeV']=(wpT_all+wsT)/(wp_all+ws)
L3['L3_mean_initial_energy_parent_primary_MeV']=wpT_all/wp_all
allC=hpC+hsC
L3['L3_initial_direction']=allC/allC.sum()
print('L3 mean all',L3['L3_mean_initial_energy_MeV'],' prim',L3['L3_mean_initial_energy_parent_primary_MeV'])
print('L3 frac scatt weight',ws/(ws+wp_all))
# ---------- L4 slowing down
# Q0 source from photon electrons (per cm^3 per MeV, per Phi0)
Tg=np.exp(np.linspace(np.log(0.01),np.log(1.2),900))
edges=np.linspace(0,1.4,701); Ecen=0.5*(edges[1:]+edges[:-1])
W_C=SM['Q0_w']*F   # per energy bin (scattered)
W_PE=SM['Q0_wpe']*F
# primary Compton lines as delta-E sources
Wprim_C=np.array([phi[i]*mu_inc(LN[i]) for i in range(2)])
Wprim_PE=np.array([phi[i]*mu_pe(LN[i]) for i in range(2)])
def pT_given_E(T,E):
    k=E/mc2
    eps=1-T/E
    c=1-(1/eps-1)/k; s2=1-c*c
    h=eps+1/eps-s2
    Z=kn_total(E)*k/(np.pi*re**2)
    out=np.where((eps>1/(1+2*k))&(eps<1),h/(Z*E),0.0)
    return out
Q0=np.zeros_like(Tg)
mask=W_C>0
Ecs=Ecen[mask]; Wcs=W_C[mask]
for Ei,Wi in zip(Ecs,Wcs):
    Q0+=Wi*pT_given_E(Tg,Ei)
for i in range(2):
    Q0+=Wprim_C[i]*pT_given_E(Tg,LN[i])
# PE sources at T=E: add as spike via bin-to-grid mapping
def add_spike(Q,Tvals,Wvals):
    for Ti,Wi in zip(Tvals,Wvals):
        if Wi<=0 or Ti<Tg[0] or Ti>Tg[-1]: continue
        j=np.searchsorted(Tg,Ti)-1
        f=(Ti-Tg[j])/(Tg[j+1]-Tg[j])
        Q[j]+=Wi*(1-f)/np.gradient(Tg)[j]; Q[j+1]+=Wi*f/np.gradient(Tg)[j+1]
    return Q
Q0=add_spike(Q0,Ecen[W_PE>0],W_PE[W_PE>0])
Q0=add_spike(Q0,LN,Wprim_PE)
Q0_tot_src=Q0
# stopping powers
Sc=Scol_f(Tg); Sr=Srad_f(Tg)
def Ldisc(T):
    hi=T/2
    if hi<=DELTA_: return 0.0
    return ne*quad(lambda td: td*moller_dsig(T,td),DELTA_,hi,limit=200)[0]
DELTA_=0.01
Ld=np.array([Ldisc(t) for t in Tg])
Scont=Sr+Sc-Ld
LDelta=Sc-Ld
# integrals: I(T)=int_T^Tmax Q dT' (cumulative from top)
def cum_from_top(f):
    # trapezoid integral from T_i to Tg[-1]
    seg=0.5*(f[1:]+f[:-1])*np.diff(Tg)
    c=np.concatenate([np.cumsum(seg[::-1])[::-1],[0.0]])
    return c
# Delta production matrix: Qd(T') = int_{T>=2T'} Phi(T) ne dsig(T,T') dT
Mmat=np.zeros((Tg.size,Tg.size))
for j,Tp in enumerate(Tg):
    if Tp<DELTA_: continue
    sel=Tg>=2*Tp
    Mmat[j,:]=np.where(sel, ne*np.array([moller_dsig(T,Tp) if T>=2*Tp else 0.0 for T in Tg]),0.0)
def Qdelta_of(Phi):
    # integrate over T (log grid): use trapezoid in T
    w=np.gradient(Tg)
    return (Mmat*(Phi*w)[None,:]).sum(1)
PhiA=cum_from_top(Q0)/Scont
PhiB=np.zeros_like(Tg)
for it in range(12):
    Qd=Qdelta_of(PhiA+PhiB)
    Qd[Tg<DELTA_]=0
    PhiBn=cum_from_top(Qd)/Scont
    if np.max(np.abs(PhiBn-PhiB))<1e-6*np.max(PhiBn+1e-30): PhiB=PhiBn; break
    PhiB=PhiBn
print('iterations',it)
Phi=PhiA+PhiB
def integ(f,lo=DELTA_,hi=Tg[-1]):
    m=(Tg>=lo)&(Tg<=hi)
    return np.trapezoid(f[m],Tg[m])
Ntot_above=integ(Phi)
Nd=integ(PhiB)
print('fluence >10keV (per Phi0):',Ntot_above,' delta part:',Nd/Ntot_above)
# bins
hb=[]
for a,b in zip(EDGES_EL[:-1],EDGES_EL[1:]):
    m=(Tg>=a)&(Tg<=b)
    # finer integration via interpolation on log grid
    tt=np.linspace(a,b,200)
    ph=np.interp(tt,Tg,Phi)
    hb.append(np.trapezoid(ph,tt))
hb=np.array(hb)
print('bins sum', hb.sum(), Ntot_above)
L4={}
L4['L4_electron_fluence_per_phi0']=Ntot_above
L4['L4_electron_fluence_spectrum']=hb
L4['L4_delta_fraction']=Nd/Ntot_above
L4['L4_mean_electron_energy_MeV']=integ(Tg*Phi)/Ntot_above
# radiative to collision, with track-end approx below Delta
PhiD=np.interp(DELTA_,Tg,Phi)
num=integ(Phi*Sr)+DELTA_*PhiD*Srad_f(DELTA_)
den=integ(Phi*Sc)+DELTA_*PhiD*Scol_f(DELTA_)
L4['L4_radiative_to_collision']=num/den
pickle.dump(dict(Tg=Tg,Phi=Phi,PhiA=PhiA,PhiB=PhiB,Scont=Scont,Ld=Ld,Q0=Q0),open('/tmp/work/slow.pkl','wb'))
'''
open('/tmp/work/ana2.py','w').write(ana2)
print("written")
