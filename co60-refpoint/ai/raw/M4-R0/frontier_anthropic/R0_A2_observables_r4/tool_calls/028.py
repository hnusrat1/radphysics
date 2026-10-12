
ana4 = r'''
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np
from ana3 import *
depth_arr=np.array([0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0])
Ks_arr=SM['kcol']*F
def Kprim(z):
    z=np.asarray(z,float)
    return sum((105.0/(100.0+z))**2*0.5*np.exp(-mu_tot(LN[i])*z)*(mu_inc(LN[i])*LN[i]*kcf_f(LN[i])+mu_pe(LN[i])*LN[i]*(1-Y_T(LN[i]))) for i in range(2))
def Kscat(z):
    z=np.asarray(z,float)
    v=np.interp(z,depth_arr,Ks_arr)
    # linear extrapolation below 0.1 cm
    slope=(Ks_arr[1]-Ks_arr[0])/(depth_arr[1]-depth_arr[0])
    v=np.where(z<depth_arr[0], Ks_arr[0]+slope*(z-depth_arr[0]), v)
    return v
def Ktot(z):
    z=np.asarray(z,float)
    return np.where(z>=0, Kprim(np.maximum(z,0))+Kscat(np.maximum(z,0)), 0.0)
zf=np.linspace(0,8,8001); dz_=zf[1]-zf[0]
Kf=Ktot(zf); Kf[0]=0.0
lam=m1
def Dof(zq):
    out=[]
    for z in zq:
        m=zf<=z
        xs=z-zf[m]                       # distance from source point z' = zf to z
        kern=np.exp(-xs/lam)/lam
        out.append(np.trapezoid(Kf[m]*kern,zf[m]))
    return np.array(out)
zq=np.linspace(0.05,7,276)
Dq=Dof(zq)
i5=np.argmin(abs(zq-5.0))
imax=np.argmax(Dq)
print('lambda(cm)',lam,' dmax from kernel model',zq[imax], ' Dmax',Dq[imax], 'D5',Dq[i5])
K5=Ktot(np.array([5.0]))[0]
beta_k=Dq[i5]/K5
print('beta (kernel model)',beta_k,'K5',K5)
print('PDD5 (kernel model)', 100*Dq[i5]/Dq[imax])
print('depth dose (per Phi0 normalised to max):')
for z,dq in zip(zq[::10],Dq[::10]): print(round(z,2), round(100*dq/Dq[imax],2), round(Ktot(np.array([z]))[0]/Ktot(np.array([zq[imax]]))[0]*100,2))
L5['L5_beta']=beta_k
L5['L5_D_over_phi0_Gy_cm2']=Dq[i5]*1.602176634e-10
L5['L5_PDD5_percent']=100*Dq[i5]/Dq[imax]
'''
open('/tmp/work/ana4.py','w').write(ana4)
import sys; sys.path.insert(0,'/tmp/work')
import ana4
