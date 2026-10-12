
import sys; sys.path.insert(0,'/tmp/work')
import pickle, numpy as np, mc as MC, phys
Tc,kerma,flu,ev,Ntot=pickle.load(open('/tmp/work/comb.pkl','rb'))
VOL=MC.VOL; F0=MC.F0; norm=F0/(Ntot*VOL)
ed=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
cos_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
T=ev[:,0]; ce=ev[:,1]; w=ev[:,2]; pr=ev[:,3]>0.5; typ=ev[:,4]
print('n events',len(T),'weight sum per src', w.sum()/Ntot, 'expected interactions in V per src (C,pe,pair):',[w[typ==k].sum()/Ntot for k in [0,1,2]])
# check Compton electron weight matches interaction rate
print('C rate check', Tc['intp_C']/Ntot+Tc['ints_C']/Ntot, w[typ==0].sum()/Ntot)
def binhist(x,wt):
    h,_=np.histogram(x,bins=ed,weights=wt); return h
inr=(T>=0.01)&(T<1.4)
h_all=binhist(T[inr],w[inr]); h_p=binhist(T[inr&pr],w[inr&pr])
out={}
out['L3_initial_spectrum']=(h_all/h_all.sum()).tolist()
out['L3_initial_spectrum_parent_primary']=(h_p/h_p.sum()).tolist()
out['L3_mean_initial_energy_MeV']=float((T*w).sum()/w.sum())
out['L3_mean_initial_energy_parent_primary_MeV']=float((T*w)[pr].sum()/w[pr].sum())
cosh=np.histogram(ce,bins=cos_edges,weights=w)[0]
out['L3_initial_direction']=(cosh/cosh.sum()).tolist()
print('mean T all',out['L3_mean_initial_energy_MeV'],'prim',out['L3_mean_initial_energy_parent_primary_MeV'])
print('frac <10keV', w[T<0.01].sum()/w.sum())
print(np.round(out['L3_initial_spectrum'],3))
print(np.round(out['L3_initial_direction'],3))
pickle.dump(out,open('/tmp/work/L3.pkl','wb'))
