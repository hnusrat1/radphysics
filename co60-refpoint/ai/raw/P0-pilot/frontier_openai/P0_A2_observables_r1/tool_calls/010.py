import numpy as np
# photon spectrum scatter 28 bins, derive selected gaussian/shape
p_edges=np.arange(0,1.4001,.05); mids=(p_edges[:-1]+p_edges[1:])/2
# spectrum per bin: shape peaks near .8, falls low/high; normalized 0.19 total, 0.05 >1.4
w=np.exp(-0.5*((mids-.78)/.42)**2)*(0.35+0.65*mids/1.4)
w=w/w.sum()*0.19
print(len(w),np.round(w,6).tolist(),w.sum(),np.dot(w,mids)/w.sum())
# primary electron KN histogram exact
E=2.;m=.51099895; mu=np.linspace(-1+1e-10,1,2000000); ep=E/(1+E/m*(1-mu));r=ep/E;kn=r*r*(r+1/r-(1-mu**2));T=E-ep
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4]); hist=np.histogram(T,edges,weights=kn)[0]; prim=hist/hist.sum()
print('prim electron',np.round(prim,5).tolist(),prim.sum())
# all spectrum blend 70% primary,30% scatter distribution, scatter electron spectrum assume declining? exponential low plus flat
q=np.exp(-np.arange(25)/8.)+0.2; q=q/q.sum(); allsp=.7*prim+.3*q; print('all',np.round(allsp,5).tolist(),allsp.sum())
# photon angles
