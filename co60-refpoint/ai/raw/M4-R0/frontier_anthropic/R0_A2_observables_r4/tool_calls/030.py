
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import ana4 as A4
zf=A4.zf; Kf=A4.Kf; lam=A4.lam; Ktot=A4.Ktot
def D_at(z):
    m=zf<=z+1e-12
    kern=np.exp(-(z-zf[m])/lam)/lam
    return np.trapezoid(Kf[m]*kern,zf[m])
zg=np.arange(0.1,7.0001,0.01)
Dg=np.array([D_at(z) for z in zg])
imax=np.argmax(Dg)
D5=D_at(5.0); K5=float(Ktot(np.array([5.0]))[0])
print('dmax',zg[imax],'Dmax',Dg[imax],'D5',D5,'K5',K5,'beta',D5/K5,'PDD5',100*D5/Dg[imax])
print('sample D/K at 0.5, 1,2,5:',[D_at(z)/float(Ktot(np.array([z]))[0]) for z in [0.5,1,2,5]])
