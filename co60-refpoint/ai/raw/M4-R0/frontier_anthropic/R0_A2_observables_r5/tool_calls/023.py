
import sys, pickle; sys.path.insert(0,'/tmp/work')
import numpy as np, importlib
import mcana, mcmod, ana as A, pdd as P
from mcana import *
from mcmod import *
tal=pickle.load(open('/tmp/work/tal_run1.pkl','rb'))
zm,Ktot,Ks,Kp = P.depth_profiles(tal, tal['nh'])
for i in [0,2,4,5,6,8,10,15,20,30,40,50,60,80,99]:
    print(round(zm[i],2), round(Ktot[i]*1.602e-10*1e12,4), "scat frac", round(Ks[i]/(Ks[i]+Kp[i]),3))
print("V-based Kcol_sc/Phi0 *1e12:", tal['Kcol_sc']/tal['nh']/Phi0_src*1.602e-10*1e12, " profile scatt at 5cm bin:", Ks[50]/Phi0_src*1.602e-10*1e12)
