
import subprocess
t=r'''
import numpy as np, sim
from phys import *
from sim import *
# instrument: count first-collision depth distribution and all collisions by depth
rng2=np.random.default_rng(5)
import sim as S
depth_first=[]; depth_all=[]; ncoll=0
orig_run=None
# quick custom transport for first collisions only
n=200000
Esrc=np.where(rng2.random(n)<0.5,1.1732,1.3325)
a=rng2.uniform(-0.05,0.05,n); b=rng2.uniform(-0.05,0.05,n)
d=np.stack([a,b,np.ones(n)],1); d/=np.linalg.norm(d,axis=1)[:,None]
src=np.array([0,0,-100.])
entry=src+d*(100/d[:,2])[:,None]
mu=mu_tot(Esrc)
s=-np.log(rng2.random(n))/mu
dbox=box_exit(entry,d)
ok=s<dbox
y=entry[ok]+s[ok,None]*d[ok]
print('frac first-collision in phantom', ok.mean(), 'expected', 1-np.exp(-0.0636*30))
h,_=np.histogram(y[:,2],bins=np.arange(0,31,1.0))
print('first-collision per src photon per 1cm bin:', np.round(h/n,5))
# analytic: primary density mu*phi, phi = exp(-mu z)/(Omega (100+z)^2)  (per src photon)
zc=np.arange(0.5,30,1.0)
an=[]
for z0 in np.arange(0,30,1.0):
    zz=np.linspace(z0,z0+1,101)
    dens=0.0636*np.exp(-0.0636*zz)/(0.00998*(100+zz)**2)*np.pi*1.0  # wrong area placeholder
    an.append(0)
print('check sum of first collisions', ok.sum()/n)
'''
open('t6.py','w').write(t)
r=subprocess.run(['python','t6.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
