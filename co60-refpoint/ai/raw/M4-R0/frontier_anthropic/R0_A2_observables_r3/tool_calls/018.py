
pdd = r'''
import sys, pickle, numpy as np
sys.path.insert(0,'/tmp/work')
import phys, mc
from phys import *
R = pickle.load(open('/tmp/work/res_t1.pkl','rb'))
N=R['N']; V=mc.V_POI; PREF = mc.OMEGA*mc.DIST**2/N
CONV=1.602176634e-10
LINES=mc.LINES
mu_L = mu_tot(LINES)
zc = mc.DZ_CENT
Kdep = R['Kdep']*PREF/(np.pi*0.25)*CONV
prim = np.zeros(zc.size)
for L,mu in zip(LINES,mu_L):
    prim += 0.5*(105.0/(100.0+zc))**2*np.exp(-mu*zc)*L*(mu_inc(L)*(1-eps_mean(L))+mu_pe(L)+mu_pair(L)*max(1-1.022/L,0))*CONV
Kz = Kdep+prim
# depth grid: bins of 0.25 cm from 0.125 .. ; use fine interpolation of K(z)
zf = np.arange(0.0, 10.0001, 0.025)
Kf = np.interp(zf, zc, Kz, left=Kz[0])
# z<0 region: vacuum (no electrons); z = 0 at surface
for lam in [0.1,0.15,0.2,0.25,0.3,0.35]:
    ker = np.exp(-(zf[:,None]-zf[None,:])/lam)*(zf[None,:]<=zf[:,None])/lam
    D = (ker*Kf[None,:]).sum(1)*0.025
    imax = np.argmax(D)
    i5 = np.argmin(np.abs(zf-5.0))
    print('lam',lam,'dmax',zf[imax],'PDD5',100*D[i5]/D[imax], 'D(0.5)/Dmax', D[np.argmin(np.abs(zf-0.5))]/D[imax])
print('K profile', np.round(Kz[:8]/Kz[0],3))
'''
open('/tmp/work/pdd.py','w').write(pdd)
import subprocess
r=subprocess.run("cd /tmp/work && python pdd.py",shell=True,capture_output=True,text=True)
print(r.stdout, r.stderr[-1500:])
