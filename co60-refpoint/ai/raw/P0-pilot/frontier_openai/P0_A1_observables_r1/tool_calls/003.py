import numpy as np
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
m=.51099895
def recoil_dist(E):
 mu=np.linspace(-1,1,300001); ep=E/(1+(E/m)*(1-mu)); r=ep/E
 kn=r*r*(r+1/r-(1-mu**2)); T=E-ep
 ix=np.digitize(T,edges)-1
 valid=T>=.01
 ix=np.clip(ix,0,24)
 h=np.bincount(ix[valid],weights=kn[valid],minlength=25).astype(float)
 return h/h.sum()
h2=recoil_dist(2.0); h08=recoil_dist(.8); h12=recoil_dist(1.2)
for name,h in [('2',h2),('.8',h08),('1.2',h12)]:
 print(name, np.round(h,5).tolist(), 'mean bin',np.dot(h,(edges[:-1]+edges[1:])/2))
allsp=.72*h2+.28*h08
allsp/=sum(allsp)
print('total',np.round(allsp,5).tolist(), 'mean bin',np.dot(allsp,(edges[:-1]+edges[1:])/2))
