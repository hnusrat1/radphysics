
phys_src=open('/tmp/phys.py').read()
phys_src=phys_src.replace("for j in [1,2,3,4,5]:","for j in [1,2,3,4,5,6]:")
phys_src+=r'''
_gl_x,_gl_w=np.polynomial.legendre.leggauss(400)
def tail_fast(T, D, Ne=NeW, variant=1):
    """N * integral_D^{T/2} eps dsigma/deps d eps, vectorised over T (array)"""
    T=np.atleast_1d(T).astype(float)
    out=np.zeros_like(T)
    for n,TT in enumerate(T):
        hi=TT/2
        if D>=hi: continue
        a=np.log(D); b=np.log(hi)
        u=0.5*(b-a)*_gl_x+0.5*(b+a); wu=0.5*(b-a)*_gl_w
        eps=np.exp(u)
        out[n]=Ne*np.sum(wu*eps*eps*moller_dsde(TT,eps,variant))
    return out
'''
open('/tmp/phys.py','w').write(phys_src)
import time
t0=time.time()
import importlib,sys
sys.path.insert(0,'/tmp')
import phys
print(phys.tail_fast(np.array([0.1,1.0]),0.01), 'time',time.time()-t0)
# compare with quad for T=1
from scipy.integrate import quad
v,_=quad(lambda e: e*phys.moller_dsde(1.0,e,1), 0.01,0.5, limit=500)
print(phys.NeW*v)
