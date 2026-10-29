import numpy as np
me=.51099895
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
# KN distribution integrated with mu for incident E
def e_hist(E, weight=1):
 mu=np.linspace(-1,1,12000)
 ep=E/(1+(E/me)*(1-mu)); rat=ep/E
 kn=rat**2*(rat+1/rat-(1-mu**2))
 ke=E-ep
 return np.histogram(ke,edges,weights=kn*weight)[0], np.sum(kn*ke*weight),np.sum(kn*weight)
pr=np.zeros(25); den=0; en=0
for e in [1.1732,1.3325]:
 h,a,b=e_hist(e);pr+=h;en+=a;den+=b
pr/=pr.sum(); print('primary mean',en/den,pr)
# scatter weights w values and photon bin center. weight by scatter fluence*approx sigma; equal no sigma
w=np.array([.001,.002,.003,.004,.006,.009,.012,.016,.02,.025,.03,.035,.04,.043,.045,.045,.044,.042,.04,.038,.035,.03,.024,.017,.009,.003,.0005,.0001]); centers=np.array([.025+i*.05 for i in range(28)])
hsum=np.zeros(25); ens=0; ds=0
for e,ww in zip(centers,w):
 if e<.02: continue
 h,a,b=e_hist(e,ww); hsum+=h; ens+=a;ds+=b
sh=hsum/hsum.sum(); print('scatter electron mean',ens/ds,sh)
tot=.7*pr+.3*sh; print('total mean',.7*en/den+.3*ens/ds); print(tot,tot.sum())