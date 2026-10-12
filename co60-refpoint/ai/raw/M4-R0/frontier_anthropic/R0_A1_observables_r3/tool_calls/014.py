
ana = r'''
import sys, pickle, numpy as np
sys.path.insert(0,'/tmp/work')
import physics as P
from physics import *
d_=pickle.load(open('/tmp/work/tot.pkl','rb'))
tot=d_['tot']; Ntot=d_['N']; zpts=d_['zpts']
im=zpts.index(5.0)
SC=P.SF/Ntot
sg=np.load('/tmp/work/main5_seg.npz'); Es=sg['E']; ds=sg['d']; Ws=sg['W']*SC
rng=np.random.default_rng(99)
z0=5.0
mu_l=np.array([P.mutot(E) for E in lines])
prim_w=0.5*np.exp(-mu_l*z0)   # per line, per Phi0
out={}
# ---------- L1
prim_tot=prim_w.sum()
scat_tot=SC*tot['A_tot'][im]
out['L1_primary_fluence_per_phi0']=prim_tot
out['L1_total_fluence_per_phi0']=prim_tot+scat_tot
out['L1_scattered_fraction_fluence']=scat_tot/(prim_tot+scat_tot)
Eprim=np.sum(prim_w*lines); Escat=SC*tot['A_Ef'][im]
out['L1_scattered_fraction_energy_fluence']=Escat/(Eprim+Escat)
out['L1_mean_photon_energy_MeV']=(Eprim+Escat)/(prim_tot+scat_tot)
spec_p=np.zeros(28)
spec_p[23]=prim_w[0]; spec_p[26]=prim_w[1]
out['L1_photon_fluence_spectrum_primary']=spec_p.tolist()
out['L1_photon_fluence_spectrum_scattered']=(SC*tot['A_E'][im]).tolist()
ap=np.zeros(10); ap[9]=1.0
out['L1_angular_primary']=ap.tolist()
ac=SC*tot['A_c'][im]; out['L1_angular_scattered']=(ac/ac.sum()).tolist()
# ---------- L2
# primary interactions per line
ints_p=np.zeros(4); tr_p=np.zeros(4)
for w_,E in zip(prim_w,lines):
    ints_p+=w_*np.array([P.muC(E),P.mupe(E),P.mupair(E),P.muR(E)])
    tr_p+=w_*np.array([P.muC(E)*P.Tmean(E),P.mupe(E)*E,P.mupair(E)*(E-2*P.me),0.0])
ints_s=SC*tot['A_int'][im]; tr_s=SC*tot['A_tr'][im]
def fr(x): x=np.array(x,float); return (x/x.sum()).tolist()
names=['compton','photoelectric','pair','rayleigh']
def dct(v): return dict(zip(names,[float(a) for a in v]))
ints=ints_p+ints_s
out['L2_interaction_fractions']=dct(fr(ints))
out['L2_energy_transfer_fractions']=dct(fr(tr_p+tr_s))
out['L2_interaction_fractions_primary']=dct(fr(ints_p))
out['L2_interaction_fractions_scattered']=dct(fr(ints_s))
out['L2_scattered_share_of_interactions']=float(ints_s.sum()/ints.sum())
# ---------- electron sampling (initial electrons)
def electrons(E,dv,W):
    """returns list of (T, cos_e, weight)"""
    Ts=[];cs=[];ws=[]
    # Compton
    wc=W*muC(E)
    eps,cc=sample_KN(E,rng)
    phi=2*np.pi*rng.random(E.size)
    dout=rotate(dv,cc,phi)
    Eout=E*eps; T=E-Eout
    pe=E[:,None]*dv-Eout[:,None]*dout
    nrm=np.linalg.norm(pe,axis=1)
    Ts.append(T); cs.append(pe[:,2]/nrm); ws.append(wc)
    # photo
    wp=W*mupe(E)
    Ts.append(E.copy()); cs.append(dv[:,2]); ws.append(wp)
    # pair (e- and e+)
    wpr=W*mupair(E); avail=np.clip(E-2*me,0,None)
    u=rng.random(E.size)
    Ts.append(u*avail); cs.append(dv[:,2]); ws.append(wpr)
    Ts.append((1-u)*avail); cs.append(dv[:,2]); ws.append(wpr)
    return np.concatenate(Ts),np.concatenate(cs),np.concatenate(ws)
Ta,ca,wa=electrons(Es,ds,Ws)
# primary parent
Tp_,cp_,wp_=electrons(lines.astype(float),np.tile([0,0,1.0],(2,1)),prim_w)
# note: pair/photo/compton for primary handled by same function
Tall=np.concatenate([Ta,Tp_]); call=np.concatenate([ca,cp_]); wall=np.concatenate([wa,wp_])
Tprim=Tp_; cprim=cp_; wprim=wp_
def spec(T,w,edges):
    m=T>=edges[0]
    h,_=np.histogram(T[m],bins=edges,weights=w[m])
    return h/h.sum()
out['L3_initial_spectrum']=spec(Tall,wall,ebE).tolist()
out['L3_initial_spectrum_parent_primary']=spec(Tprim,wprim,ebE).tolist()
out['L3_mean_initial_energy_MeV']=float(np.sum(Tall*wall)/np.sum(wall))
out['L3_mean_initial_energy_parent_primary_MeV']=float(np.sum(Tprim*wprim)/np.sum(wprim))
hc,_=np.histogram(call,bins=cb,weights=wall); out['L3_initial_direction']=(hc/hc.sum()).tolist()
'''
open('/tmp/work/ana1.py','w').write(ana)
exec(open('/tmp/work/ana1.py').read())
import json
for k in ['L1_primary_fluence_per_phi0','L1_total_fluence_per_phi0','L1_scattered_fraction_fluence','L1_scattered_fraction_energy_fluence','L1_mean_photon_energy_MeV','L2_interaction_fractions','L2_energy_transfer_fractions','L2_interaction_fractions_primary','L2_interaction_fractions_scattered','L2_scattered_share_of_interactions','L3_mean_initial_energy_MeV','L3_mean_initial_energy_parent_primary_MeV']:
    print(k, out[k])
print(np.round(out['L3_initial_spectrum'],4))
print(np.round(out['L3_initial_spectrum_parent_primary'],4))
print(np.round(out['L3_initial_direction'],4))
print(np.round(out['L1_angular_scattered'],4))
print(np.round(np.array(out['L1_photon_fluence_spectrum_scattered']),4))
