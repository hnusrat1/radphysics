import numpy as np, pickle
xcom=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
E_x=xcom[:,0]; mu_coh=xcom[:,1]; mu_inc=xcom[:,2]; mu_pe=xcom[:,3]; mu_pair=xcom[:,4]+xcom[:,5]; mu_tot=xcom[:,6]
LE=np.log(E_x)
def mut(E,mu):
    mu=np.maximum(mu,1e-30)
    return np.exp(np.interp(np.log(np.clip(E,1e-6,20)), LE, np.log(mu)))
with open('/tmp/photon_results.pkl','rb') as f: res=pickle.load(f)
fl_prim=res['fl_prim']; fl_scat=res['fl_scat']
scale=0.72948169572262/fl_prim.sum()
spec_p=fl_prim*scale; spec_s=fl_scat*scale
Edges=np.linspace(0,1.4,29); Ec=(Edges[:-1]+Edges[1:])/2
spec=spec_p+spec_s
# interaction-weighted fractions
def fracs(phi,Epts):
    phi=np.asarray(phi,float)
    mc=mut(Epts,mu_coh); mi=mut(Epts,mu_inc); mp=mut(Epts,mu_pe); mpr=mut(Epts,mu_pair); mt=mut(Epts,mu_tot)
    w_c=phi*mi; w_r=phi*mc; w_p=phi*mp; w_pr=phi*mpr; w_t=phi*mt
    return np.array([w_r.sum(),w_c.sum(),w_p.sum(),w_pr.sum()])/w_t.sum(), (w_t.sum(),)
print('L2 interaction fractions (fluence-weighted):', fracs(spec,Ec)[0])
# primary only (two lines)
E1,E2=1.1732,1.3325; f1,f2=spec_p[23],spec_p[26]
print('prim spectrum bin values', f1,f2)
fr_prim = fracs(np.array([f1,f2]), np.array([E1,E2]))[0]
print('L2 interaction fractions primary:', fr_prim)
# scattered only
fr_scat = fracs(spec_s,Ec)[0]
print('L2 interaction fractions scattered:', fr_scat)
# energy transfer fractions
# compton avg energy transfer fraction: (E-<E'>)/E ; compute KN mean
def compton_etr_frac(E):
    E=np.atleast_1d(E).astype(float); out=np.zeros(len(E))
    # integrate over KN numerically on cos grid
    for i,e in enumerate(E):
        alpha=e/0.511
        mu=np.linspace(-1,1,2000)
        d=1+alpha*(1-mu)
        p=((1+mu*mu)/2+(alpha**2*(1-mu)**2)/(2*d))/(d*d)
        Ep=e/d
        Etr=e-Ep
        out[i]=(p*Etr).sum()/(p*e).sum()*1 if False else (p*Etr).sum()/(p*e).sum()
    return out
# energy transfer per interaction by process: compton: E*frac; pe: E; pair: E-1.022
def etr_fracs(phi,Epts):
    phi=np.asarray(phi,float)
    mc=mut(Epts,mu_coh); mi=mut(Epts,mu_inc); mp=mut(Epts,mu_pe); mpr=mut(Epts,mu_pair); mt=mut(Epts,mu_tot)
    cf=compton_etr_frac(Epts)
    w_c=phi*mi*Epts*cf
    w_p=phi*mp*Epts
    w_pr=phi*mpr*np.maximum(Epts-1.022,0)
    tot=w_c.sum()+w_p.sum()+w_pr.sum()
    return np.array([0.0,w_c.sum(),w_p.sum(),w_pr.sum()])/tot
print('L2 energy transfer fractions:', etr_fracs(spec,Ec))
# also check compton etr frac at 1.25
print('KN etr frac at 1.25:', compton_etr_frac(np.array([1.25]))[0])