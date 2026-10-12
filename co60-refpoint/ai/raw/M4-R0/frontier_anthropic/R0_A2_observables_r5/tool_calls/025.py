
import sys, pickle; sys.path.insert(0,'/tmp/work')
import numpy as np, importlib
import mcana, mcmod, ana as A, pdd as P, final as F, kern as K
from mcana import *
from mcmod import *
tal=pickle.load(open('/tmp/work/tal_run1.pkl','rb'))
rng=np.random.default_rng(7)
ptal,Pper=A.primary_electron_tally(rng, Ns=400000)
nh=tal['nh']
Cnc = tal['q_fine_all_cos']/nh + ptal['q_fine_all_cos']
dF = FINE[1:]-FINE[:-1]; cF=np.sqrt(FINE[:-1]*FINE[1:])
xg, kr, tot = K.kernel_from_sources(Cnc, cF, dF, zbin=0.1, xmin=-2.0, xmax=12.0, ns=400)
kn = kr*0.1/ (kr*0.1).sum()     # fraction per 0.1 cm bin
print("kernel total", (kr*0.1).sum(), "mean depth", np.sum(kn*(xg[:-1]+0.05)))
# depth-dose
zm,Ktot,Ks,Kp = P.depth_profiles(tal, nh)
zgrid = np.arange(0,10,0.1)+0.05   # bins 0.1 wide
Kz = Ktot   # per bin 0..10 (100 bins)
# convolution D(z_i) = sum_j K(z_j) * kn[(i-j)] where kn index mapping x=z_i-z_j
xc = xg[:-1]+0.05
D = np.zeros(100)
for i in range(100):
    s=0.0
    for j in range(100):
        xk = zm[i]-zm[j]
        k_idx = int(np.floor((xk - xg[0])/0.1))
        if 0<=k_idx<len(kn): s+=Kz[j]*kn[k_idx]
    D[i]=s
imax=np.argmax(D)
print("dmax approx", zm[imax], "D(5)/Dmax*100", 100*D[np.argmin(np.abs(zm-5))]/D.max())
print(np.round(D[:12]/D.max(),3))
