import numpy as np
from scipy.interpolate import interp1d
estar=np.loadtxt('/tmp/data/estar_water.txt',comments='#'); mid=(np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])[:-1]+np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])[1:])/2
Fl=np.array([.000256791710,.000595855103,.000785214003,.000898525445,.000958434306,.000987112919,.000994671708,.000984757320,.000960237512,.000924048525,.000880567440,.000834151186,.000783745512,.000729109869,.000668194714,.000598997572,.000517864564,.000423951396,.000310245363,.000207308053,.000139978496,.0000721316856,.0000169255597,.000000078866,0])
sc=np.exp(interp1d(np.log(estar[:,0]),np.log(estar[:,1]),fill_value='extrapolate')(np.log(mid)))
print('collision power',sum(Fl*sc),'rad',sum(Fl*np.exp(interp1d(np.log(estar[:,0]),np.log(estar[:,2]),fill_value='extrapolate')(np.log(mid)))),'mean',sum(Fl*mid)/sum(Fl))