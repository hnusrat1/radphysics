import numpy as np
estar=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
# make flu array from prior? recreate base/full values
flu=np.array([0.001254305532225,0.0009070962554083333,0.000727587734,0.000801882005,0.000834810967,0.000842680572,0.000834392414,0.000809766338,0.000774126844,0.000735726138,0.000692417878,0.00065140986,0.000605090221,0.000547918393,0.000496573813,0.000434431692,0.000367020566,0.000294398844,0.000192989341,0.000129550863,8.24316951e-05,2.46931706e-05,0,0,0])
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4]);mid=(edges[:-1]+edges[1:])/2
sc=np.exp(np.interp(np.log(mid),np.log(estar[:,0]),np.log(estar[:,1])))
sr=np.exp(np.interp(np.log(mid),np.log(estar[:,0]),np.log(estar[:,2])))
print(sum(flu*sr)/sum(flu*sc),sum(flu*sr),sum(flu*sc))
