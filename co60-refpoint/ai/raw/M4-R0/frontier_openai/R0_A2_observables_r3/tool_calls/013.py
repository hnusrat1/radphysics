import numpy as np
m=.51099895
# edges reused, photon v and Ebin from above
hist_all=np.zeros(25);hist_p=np.zeros(25); hist_sc=np.zeros(25); meanp=meant=0;countp=countsc=0
# numerical electron response normalized KN and estimated incoherent cross section from XCOM
cs_interp=lambda E,j: np.exp(np.interp(np.log(E),logx,np.log(np.maximum(x[:,j],1e-300))))
cc=np.linspace(-1,1,20001); dcc=cc[1]-cc[0]
def recoil_hist(E, weight, hist):
 ep=1/(1+E/m*(1-cc)); ds=ep**2*(ep+1/ep-(1-cc**2))/2
 T=E*(1-ep); ww=ds*dcc; ww[[0,-1]]*=.5
 norm=np.sum(ww)
 mask=T>=.01
 hist+=np.histogram(T[mask],bins=edges,weights=weight*ww[mask]/norm)[0]
 return weight*np.sum(T[mask]*ww[mask])/norm, weight*np.sum(ww[mask])/norm
for E in [1.1732,1.3325]:
 w=.3645*cs_interp(E,2)
 mm,ct=recoil_hist(E,w,hist_p);meanp+=mm;countp+=ct
for E,f in zip(Ebin,v):
 w=f*cs_interp(E,2)
 mm,ct=recoil_hist(E,w,hist_sc);meant+=mm;countsc+=ct
# primary sources hist counts are weighted actual interactions, scatter interaction rates; normalized overall
print('counts',countp,countsc,'scatter parent fraction',countsc/(countp+countsc))
print('mean p',meanp/countp,'mean sc',meant/countsc)
qall=hist_p+hist_sc
print('means',qall@((edges[:-1]+edges[1:])/2)/qall.sum())
print((qall/qall.sum()).tolist())
print('p norm',(hist_p/hist_p.sum()).tolist())