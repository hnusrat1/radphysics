import numpy as np
estar=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
def stop(E,col=1):return np.exp(np.interp(np.log(E),np.log(estar[:,0]),np.log(estar[:,col])))
Eedges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4]);mid=(Eedges[:-1]+Eedges[1:])/2
bp=np.array([272,356,322,342,301,306,288,314,321,308,303,287,288,355,324,383,417,456,636,395,293,363,156,0,0],float)
bs=np.array([2473,1313,652,430,331,242,195,164,146,113,107,75,84,71,48,47,37,23,20,13,8,3,0,0,0],float)
ref=171441.905976
adj=(bp+bs)/ref*((Eedges[1:]-Eedges[:-1])*.5)/np.array([stop(x) for x in mid])
print(adj,adj.sum())
