import importlib, sys, time, numpy as np
sys.path.insert(0,'/tmp')
import photonmc
importlib.reload(photonmc)
pd_depths=[0.2,0.35,0.5,0.65,0.8,1.0,1.5,2,3,4,5,6,8,10]
res = photonmc.photon_mc(300000, seed=5, verbose=False, save_src=False, pd_depths=pd_depths)
print('int_all',res['int_all'])
print('Kdep', res['Kdep'])
print('Ndep', res['Ndep'])