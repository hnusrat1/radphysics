import numpy as np
# electron grid, KN binned distribution from above h normalized
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
q=np.array([0.03467654,0.0427149,0.04204434,0.04141542,0.04083238,0.04031258,0.03986744,0.0395205,0.0392957,0.03923144,0.0393836,0.0398237,0.04066218,0.04207078,0.04431134,0.04782442,0.0533822,0.06244288,0.07806276,0.04732262,0.03624142,0.04719832,0.02136254,0,0])
# stop data
estar=np.loadtxt('/tmp/data/estar_water.txt',comments='#');
# get primary spectrum track fluence per bins via midpoint survival cumulative/S
mid=(edges[:-1]+edges[1:])/2
S=np.interp(mid,estar[:,0],estar[:,3])
# estimate survival using initial energies bin-center, improve cap
surv=np.array([q[mid>=x].sum() for x in mid])
raw=surv/S*np.diff(edges)
print('raw',raw/raw.sum())
# delta ray distribution choose spectrum ~1/T generation times range, rough extra in bins declining with E
# Moller dN/dT ~1/T^2, track range approximately T^1.5 at low energy -> track weights ~T^-0.5 per T; bins integral ~ width/sqrt(E)
delta=np.diff(edges)/np.sqrt(mid)
delta[mid>0.5]*=0.3 # tail limited
# compare
print('delta norm',delta/delta.sum())
allraw=.7*raw/raw.sum()+.3*delta/delta.sum()
print('all',allraw.tolist(), 'mean',np.dot(allraw,mid), 'binsum',allraw.sum())