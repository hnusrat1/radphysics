
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import ana4 as A4
zf=A4.zf; Kf=A4.Kf; lam=A4.lam
print(lam, zf[4000], Kf[4000], Kf[3900], Kf[4100])
z=5.0
m=zf<=z
kern=np.exp(-(z-zf[m])/lam)/lam
print('norm of kernel', np.trapezoid(kern,zf[m]))
print('D5',np.trapezoid(Kf[m]*kern,zf[m]), 'K5', Kf[4000])
print(A4.Ktot(np.array([4.9,5.0,5.1])))
