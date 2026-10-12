import numpy as np
# 50m results
prim=np.array([478,554,574,544,557,529,488,486,519,526,527,519,507,519,562,605,739,776,1034,611,477,579,268,1,0.])
sc=np.array([237,264,253,257,244,202,213,186,214,191,192,175,211,189,207,239,274,287,382,210,161,202,89,0,0.])
print(prim.sum(),sc.sum(),np.dot(prim,np.arange(25)),np.dot(sc,np.arange(25)))
print('means hist bin mids')
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
mids=(edges[:-1]+edges[1:])/2
print((prim@mids)/prim.sum(),(sc@mids)/sc.sum(), ((prim+sc)@mids)/(prim.sum()+sc.sum()))
# MC counts and interaction process order
ints=np.array([[13070,5144],[1,0],[3,0],[5,0]],float)
trs=np.array([[7672.31758,2846.729135],[0,0],[0,0],[0,0]],float)
print(ints.sum(), ints.sum(axis=1)/ints.sum(),ints[:,0]/ints[:,0].sum(),ints[:,1]/ints[:,1].sum(),ints[:,1].sum()/ints.sum())
print('energy frac',trs[:,1].sum()/(trs[:,0].sum()+trs[:,1].sum()))
print('primary elec mean',7673.906637/12979,'sc mean',2846.387891/5079,'all', (7673.906637+2846.387891)/(12979+5079))