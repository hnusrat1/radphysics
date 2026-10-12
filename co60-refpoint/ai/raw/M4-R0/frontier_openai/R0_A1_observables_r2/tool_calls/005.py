import numpy as np
prim=np.zeros(28); prim[23]=.364;prim[26]=.364
w=np.array([.005,.01,.02,.035,.05,.06,.06,.055,.05,.045,.04,.04,.04,.04,.04,.04,.04,.04,.04,.04,.04,.04,.04,.04,.035,.03,.025,.015]); scat=.1*w/w.sum()
E=np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4]); centers=(E[:-1]+E[1:])/2
pmean=(prim@centers)/prim.sum(); smean=scat@centers/.1
print('pm sm totalmean',pmean,smean,(prim@centers+scat@centers)/.828)
print('scat energy fraction',scat@centers/(prim@centers+scat@centers))
# L2 aggregation
P={'compton':.9987,'photoelectric':.0003,'pair':.0009,'rayleigh':.0001}
S={'compton':.995,'photoelectric':.004,'pair':.0002,'rayleigh':.0008}
f=.14
print({k:(1-f)*P[k]+f*S[k] for k in P})
# L3 arrays manually get from previous replicate computation imports omitted
