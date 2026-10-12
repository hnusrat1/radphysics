
code = r'''
import numpy as np, time, json
rng=np.random.default_rng(12345)
re=2.8179403e-13; ne=3.3428e23; me=0.51099895
def sigma_KN(E):
    k=E/me
    t1=(1+k)/k**2*(2*(1+k)/(1+2*k)-np.log(1+2*k)/k)
    t2=np.log(1+2*k)/(2*k)-(1+3*k)/(1+2*k)**2
    return 2*np.pi*re**2*(t1+t2)
def mu_parts(E):
    sKN=ne*sigma_KN(E)
    coh=sKN*0.032*(E/0.1)**(-1.2)
    pe=0.0028*(E/0.1)**(-3.0)
    pair=np.where(E>1.022, 0.0005*np.clip((E-1.022)/0.978,0,None)**2, 0.0)
    return sKN,coh,pe,pair
mm=np.linspace(-1,1,4001)
for E in [0.02,0.1,0.5,1.1732,1.3325]:
    k=E/me; eps=1/(1+k*(1-mm)); f=eps**2*(eps+1/eps-(1-mm**2)); print('fmax',E,f.max())
'''
open('mc_base.py','w').write(code)
import subprocess
print(subprocess.run(['python','mc_base.py'],capture_output=True,text=True).stdout)
