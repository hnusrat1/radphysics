import numpy as np
from scipy.integrate import quad
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
cedges=np.linspace(-1,1,11)
me=.51099895
# integrate sampled mu over KN for primary electron energy + angles
for E in [1.1732,1.3325]:
 mu=np.linspace(-1,1,200000)
 ep=E/(1+(E/me)*(1-mu))
 ratio=ep/E
 w=ratio**2*(ratio+1/ratio-(1-mu**2)) # d sigma/d mu proportional
 ke=E-ep
 ce=(E-ep*mu)/np.sqrt(np.maximum(E*E+ep*ep-2*E*ep*mu,1e-25))
 hist=np.histogram(ke,edges,weights=w)[0]; ang=np.histogram(ce,cedges,weights=w)[0]
 print(E, hist/hist.sum(),ang/ang.sum(),np.sum(ke*w)/w.sum())