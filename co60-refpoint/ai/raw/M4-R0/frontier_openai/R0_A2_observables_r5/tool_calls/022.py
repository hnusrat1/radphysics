import numpy as np
prim=np.array([307,329,333,321,340,307,291,305,319,315,322,301,294,312,368,357,452,471,618,360,283,359,160,1,0.],float)
sc=np.array([141,156,144,151,151,125,132,116,120,127,114,105,133,108,106,141,153,171,224,129,94,131,53,0,0.],float)
initial=prim+sc
estar=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
# interpolate log-log
from scipy.interpolate import interp1d
Etab=estar[:,0]
S=interp1d(np.log(Etab),np.log(estar[:,3]),fill_value='extrapolate')
edge=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
mid=(edge[:-1]+edge[1:])/2
s=np.exp(S(np.log(mid)))
N=30e6; Omega=0.00997507; factor=Omega*105**2/(N*np.pi*.2)
# electrons track per bin from local source by slowing down
Fl=[]
for i in range(len(mid)):
    full=initial[i+1:].sum()
    Fl.append(factor*(full+0.5*initial[i])*(edge[i+1]-edge[i])/s[i])
Fl=np.array(Fl)
print(factor,Fl.sum(),Fl,Fl/Fl.sum())
print('mean',np.sum(Fl*mid)/Fl.sum())
