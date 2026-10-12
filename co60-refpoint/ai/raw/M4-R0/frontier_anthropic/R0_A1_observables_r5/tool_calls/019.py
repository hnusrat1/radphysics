
code = r'''
import numpy as np, pickle, sys
sys.path.insert(0,'/tmp/work')
import mc, phys, ea
from phys import me, re, ne_w as ne, ne_air
from sim import E_LINES
from scipy import integrate
seg,hist,NTOT=pickle.load(open('/tmp/work/tallies.pkl','rb'))
V=np.pi*1.0**2*0.2
R=mc.Omega*105.0**2/(V*NTOT)
E=seg['E']; tl=seg['tl']; prim=seg['prim'].astype(bool); ux,uy,uz=seg['ux'],seg['uy'],seg['uz']
u=np.c_[ux,uy,uz]
rng=np.random.default_rng(5)
muC=mc.mu_c(E); mupe=mc.mu_pe(E); mupair=mc.mu_pair(E)
w=tl*R
Ee_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
cb=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
M=8
# ---- Compton electrons: sample M per segment
Ts=[];Ws=[];Cs=[];Ps=[]; wsC=[]; 
idx=np.arange(E.size)
Eall=np.repeat(E,M); uall=np.repeat(u,M,axis=0); pr_all=np.repeat(prim,M)
wC_all=np.repeat(tl*muC*1.0,M)/M      # for L3 weights (no R)
wC_R=np.repeat(w*muC,M)/M             # for production (per Phi0)
eps,c=sample_KN=None,None
from mc import sample_KN, rotate
eps,c=sample_KN(Eall,rng)
phi=rng.uniform(0,2*np.pi,Eall.size)
uout=rotate(uall,c,phi)
Tc=Eall*(1-eps)
pe_vec=Eall[:,None]*uall-(Eall*eps)[:,None]*uout
pe_norm=np.linalg.norm(pe_vec,axis=1)
cos_e=pe_vec[:,2]/pe_norm
# photoelectrons
Eph=E; wPE=tl*mupe; wPE_R=w*mupe
cphe=np.empty(E.size)
todo=np.arange(E.size)
while todo.size:
    cc=rng.uniform(-1,1,todo.size); rr=rng.random(todo.size)
    acc=rr*1.0<(1-cc*cc)
    cphe[todo[acc]]=cc[acc]; todo=todo[~acc]
phi2=rng.uniform(0,2*np.pi,E.size)
u_pe=rotate(u,cphe,phi2)
# pair: two electrons with T=(E-2m)/2 along photon direction
Tp=(E-2*me)/2
mask_pair=mupair>0
# assemble electron list for L3
T_all=np.concatenate([Tc, Eph, Tp[mask_pair], Tp[mask_pair]])
W_all=np.concatenate([wC_all, wPE, (tl*mupair)[mask_pair], (tl*mupair)[mask_pair]])
PR_all=np.concatenate([pr_all, prim, prim[mask_pair], prim[mask_pair]])
COS_all=np.concatenate([cos_e, cphe, uz[mask_pair], uz[mask_pair]])
# photo-electron cos wrt beam axis (use u_pe)
COS_all[Tc.size:Tc.size+E.size]=u_pe[:,2]
out={}
edges=Ee_edges
sel=T_all>=0.01
h,_=np.histogram(T_all[sel],bins=edges,weights=W_all[sel]); out['L3_initial_spectrum']=(h/h.sum()).tolist()
h,_=np.histogram(T_all[sel&PR_all],bins=edges,weights=W_all[sel&PR_all]); out['L3_initial_spectrum_parent_primary']=(h/h.sum()).tolist()
out['L3_mean_initial_energy_MeV']=float((T_all*W_all).sum()/W_all.sum())
out['L3_mean_initial_energy_parent_primary_MeV']=float((T_all*W_all)[PR_all].sum()/W_all[PR_all].sum())
h,_=np.histogram(COS_all[sel],bins=cb,weights=W_all[sel]); out['L3_initial_direction']=(h/h.sum()).tolist()
print('L3 mean all',out['L3_mean_initial_energy_MeV'],'prim',out['L3_mean_initial_energy_parent_primary_MeV'])
print('frac T<0.01 of photon electrons', W_all[~sel].sum()/W_all.sum())
# ---- production densities per Phi0 (per cm3 per MeV)
dT=0.001
Tg=np.arange(0.0005,1.4+1e-9,dT)
qC_samp=np.histogram(Tc,bins=np.r_[Tg-dT/2,Tg[-1]+dT/2],weights=wC_R)[0]/dT
qPE=np.histogram(Eph,bins=np.r_[Tg-dT/2,Tg[-1]+dT/2],weights=wPE_R)[0]/dT
qP=np.histogram(Tp[mask_pair],bins=np.r_[Tg-dT/2,Tg[-1]+dT/2],weights=(w*mupair)[mask_pair])[0]/dT*2
qp=qC_samp+qPE+qP
print('production integrals (per Phi0 cm-1):', qC_samp.sum()*dT, qPE.sum()*dT, qP.sum()*dT)
# stopping powers
Scol=np.array([ea.S_col_bethe(t) for t in Tg]); Srad=Scol*0.0075*Tg
Stot=Scol+Srad
mask10=Tg>=0.01
# Mollers matrix eps x T
Tgrid=Tg[mask10]; wT=dT
Mmat=np.zeros((Tg.size,Tgrid.size))
for j,T in enumerate(Tgrid):
    e=Tg
    x=e/T
    ok=(e>=0.01)&(e<=T/2)
    val=np.zeros_like(e)
    val[ok]=ea.dsig_deps(T,e[ok])
    Mmat[:,j]=val
def cum_above(q):
    # sum_{T'>T} q dT' (using bin values, excluding own bin midpoint half)
    c=np.cumsum(q[::-1])[::-1]*dT
    return c - q*dT/2
# iterate
Phi=np.zeros(Tg.size)
Phip=np.zeros(Tg.size)
Phip[mask10]=cum_above(qp)[mask10]/Stot[mask10]
Phi=Phip.copy()
for it in range(40):
    qd=ne*(Mmat@(Phi[mask10]*dT))
    qd[Tg<0.01]=0
    Phi_new=np.zeros(Tg.size)
    Phi_new[mask10]=cum_above(qp+qd)[mask10]/Stot[mask10]
    diff=np.max(np.abs(Phi_new-Phi))/np.max(Phi)
    Phi=Phi_new
    if diff<1e-6: break
print('iterations',it,diff)
Phi_p=Phip
Phi_d=Phi-Phip
Phi10=Phi*mask10
tot_fl=(Phi10).sum()*dT
out['L4_electron_fluence_per_phi0']=float(tot_fl)
out['L4_delta_fraction']=float(Phi_d.sum()/Phi.sum())
out['L4_mean_electron_energy_MeV']=float((Tg*Phi10).sum()/Phi10.sum())
# spectrum
ids=np.digitize(Tg,Ee_edges)-1
spec=[float((Phi10[(Tg>=Ee_edges[i])&(Tg<Ee_edges[i+1])]).sum()*dT) for i in range(25)]
out['L4_electron_fluence_spectrum']=spec
out['L4_radiative_to_collision']=float((Srad*Phi10).sum()/(Scol*Phi10).sum())
# radiated energy and dose
Krad_tot = float((Srad*Phi10).sum()*dT)
Scol_fl=float((Scol*Phi10).sum()*dT)
pickle.dump(dict(out=out,Phi=Phi,Tg=Tg,Scol=Scol,Srad=Srad,Krad=Krad_tot,Scol_fl=Scol_fl,Phi_d=Phi_d),open('/tmp/work/part2.pkl','wb'))
print('L4 total',tot_fl,'delta frac',out['L4_delta_fraction'],'mean T',out['L4_mean_electron_energy_MeV'],'rad/col',out['L4_radiative_to_collision'])
print('spec',np.round(spec,5))
print('Krad (MeV/cm, per Phi0)',Krad_tot,' Scol*Phi (>=10keV)',Scol_fl)
'''
open('/tmp/work/an2.py','w').write(code)
o,e=runpy('/tmp/work/an2.py'); print(o[-5000:], e[-3000:])
