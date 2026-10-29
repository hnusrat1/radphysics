import numpy as np
# KN sample scatter theta for incident 2 MeV
E=2.; m=.51099895
mu=np.linspace(-1,1,1000001)
ep=E/(1+E/m*(1-mu)); ratio=ep/E
# d sigma / dOmega proportional ratio^2*(ratio+1/ratio-(1-mu**2))
w=ratio**2*(ratio+1/ratio-(1-mu**2))
# recoil energy and angle
t=E-ep
pz=E-ep*mu; pperp=ep*np.sqrt(1-mu**2); ca=pz/np.sqrt(pz**2+pperp**2)
# hist energies and angular
eedges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
cosedges=np.linspace(-1,1,11)
for x in [np.histogram(t,edges,weights=w)[0], np.histogram(ca,cosedges,weights=w)[0]]:
 print(np.round(x/x.sum(),4))
print('meanT',np.trapz(w*t,mu)/np.trapz(w,mu), 'max',t.max(), 'above1.4fraction',np.trapz(w*(t>1.4),mu)/np.trapz(w,mu))