import numpy as np
np.random.seed(0)
fine=np.arange(0.01,1.401,0.005)
Phi=np.random.rand(len(fine))
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
tot=0
for b in range(25):
    m=(fine>=el_edges[b])&(fine<el_edges[b+1])
    tot+=np.trapezoid(Phi[m],fine[m])
print('sum bins', tot, 'direct', np.trapezoid(Phi,fine))