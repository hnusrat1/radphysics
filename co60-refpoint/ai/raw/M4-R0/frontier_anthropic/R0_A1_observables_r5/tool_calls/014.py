
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import phys
re=phys.re; me=phys.me; ne=phys.ne_w; alpha=1/137.036
def beta2(T):
    tau=T/me; return tau*(tau+2)/(tau+1)**2
def S_col_bethe(T,I=75e-6,n=ne):
    tau=T/me; b2=beta2(tau*me)
    F=(1-b2)*(1+tau**2/8-(2*tau+1)*np.log(2))
    L=np.log(tau**2*(tau+2)/(2*(I/me)**2))
    return 2*np.pi*re**2*me*n/b2*(L+F)
def S_rad(T,n=ne):
    # Bethe-Heitler complete screening approx, per atom Z(Z+eta) ~ Z^2 (use Z(Z+1))
    E=T+me
    wH,wO=0.111894,0.888106
    tot=0
    for w,Z,A in [(wH,1,1.00794),(wO,8,15.9994)]:
        tot+= w*phys.NA/A*4*alpha*re**2*Z*(Z+1)*E*(np.log(183*Z**(-1/3.))+ 1/18.)
    return tot
for T in [0.01,0.1,0.5,1.0,10.0]:
    print(T, S_col_bethe(T), S_rad(T))
