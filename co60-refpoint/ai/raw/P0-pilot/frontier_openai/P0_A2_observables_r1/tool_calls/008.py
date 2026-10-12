import numpy as np
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4]); mid=(edges[:-1]+edges[1:])/2
estar=np.loadtxt('/tmp/data/estar_water.txt'); sc=np.interp(mid,estar[:,0],estar[:,1]); width=np.diff(edges)
# initial sources assumed energy uniform 0-1.6, cumulative births > midpoint, track spectrum
cum=np.maximum(1-mid/1.6,0)
w=width*cum/sc
print(np.round(w/w.sum(),4), 'meanE',sum(w*mid)/sum(w),'Smean',sum(w*sc)/sum(w))