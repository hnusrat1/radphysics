
tables_code = r'''
import numpy as np
from phys import *
EGRID=np.exp(np.linspace(np.log(0.005),np.log(1.4),400))
def Y_T(T):
    T=np.asarray(T,float)
    return np.interp(np.log(np.maximum(T,0.01)),np.log(ES[:,0]),ES[:,5])
def R_T(T):
    T=np.asarray(T,float)
    return np.exp(np.interp(np.log(np.maximum(T,1e-4)),np.log(ES[:,0]),np.log(ES[:,4])))
def build():
    ftr=[];kcf=[];gyf=[]
    for E in EGRID:
        k=E/mc2; eps0=1/(1+2*k)
        eg=np.exp(np.linspace(np.log(eps0),0,4000))
        c=1-(1/eg-1)/k; s2=1-c*c
        h=eg+1/eg-s2
        Z=np.trapezoid(h,eg)
        f=h/Z
        T=E*(1-eg)
        ftr.append(np.trapezoid(f*(1-eg),eg))
        kcf.append(np.trapezoid(f*(1-eg)*(1-Y_T(T)),eg))   # collisional energy fraction per interaction
        gyf.append(np.trapezoid(f*(1-eg)*Y_T(T),eg))
    ftr=np.array(ftr);kcf=np.array(kcf);gyf=np.array(gyf)
    np.savez('/tmp/work/tables.npz',E=EGRID,ftr=ftr,kcf=kcf,gyf=gyf)
    return ftr,kcf,gyf
if __name__=='__main__':
    ftr,kcf,gyf=build()
    print(ftr[::80]); print(kcf[::80]); print(gyf[::80])
'''
open('/tmp/work/tables.py','w').write(tables_code)
import sys; sys.path.insert(0,'/tmp/work')
import tables
ftr,kcf,gyf=tables.build()
print(tables.EGRID[::80]); print(ftr[::80]); print(kcf[::80]); print(gyf[::80])
