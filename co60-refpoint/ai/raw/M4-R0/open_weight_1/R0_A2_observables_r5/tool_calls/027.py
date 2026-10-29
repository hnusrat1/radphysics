# compute g from L3 initial spectrum using ESTAR radiation yield
import numpy as np, pickle
estar=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
eT=estar[:,0]; Y=estar[:,5]
with open('/tmp/photon_results.pkl','rb') as f: res=pickle.load(f)
Te=res['L3_Te']
Yat=np.interp(Te, eT, Y)
g = (Te*Yat).sum()/Te.sum()
print('g from L3 spectrum (3300 electrons):', g)
print('mean Y weighted', (Te*Yat).sum()/Te.sum(), 'mean Y unweighted', Yat.mean())
# also primary-only g
p=res['L3_parent']==1
g_prim=(Te[p]*np.interp(Te[p],eT,Y)).sum()/Te[p].sum()
print('g primary parent:', g_prim)