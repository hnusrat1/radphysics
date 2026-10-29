
import sys, pickle; sys.path.insert(0,'/tmp/work')
import numpy as np, xs, mell
from elec import EB25
cedges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0001])
tot=pickle.load(open('/tmp/work/tot.pkl','rb')); res=pickle.load(open('/tmp/work/eres1.pkl','rb'))
N=tot['N']; VC=np.pi*0.2; PHI=1/(0.0099751*110**2); norm=N*VC*PHI
B=tot['births']; T0=B[:,0]; c=B[:,1]; F=B[:,2]; w=B[:,3]; kind=B[:,4]
Y=xs.Yld(np.maximum(T0,1e-2))*(T0>=0.01)
inb=(T0>=0.01)&(T0<1.4)
h=np.histogram(T0[inb],bins=EB25,weights=w[inb])[0]; L3spec=h/h.sum()
mE=np.sum(w*T0)/np.sum(w)
p=F==0
h0=np.histogram(T0[inb&p],bins=EB25,weights=w[inb&p])[0]; L3spec_p=h0/h0.sum()
mEp=np.sum((w*T0)[p])/np.sum(w[p])
cb=np.histogram(c,bins=cedges,weights=w)[0]; L3dir=cb/cb.sum()
print('L3 spectrum',np.round(L3spec,4)); print('mean',mE,'prim',mEp)
print('L3 spec prim',np.round(L3spec_p,4)); print('dir',np.round(L3dir,4))
print('birth fractions by kind', [w[kind==k].sum()/w.sum() for k in range(4)], 'scat frac births', w[F==1].sum()/w.sum())
Kt=np.sum(w*T0)/norm; Kc=np.sum(w*T0*(1-Y))/norm
print('K',Kt,'Kcol',Kc,'g',1-Kc/Kt)
dose_scat=np.sum((w*T0*(1-Y))[F==1])/np.sum(w*T0*(1-Y))
print('dose frac scattered parent', dose_scat)
print('frac births T0<1.4 and >=0.01', w[inb].sum()/w.sum(), 'T0 max', T0.max())
