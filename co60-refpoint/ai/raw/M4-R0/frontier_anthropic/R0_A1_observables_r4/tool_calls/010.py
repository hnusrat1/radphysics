
tab = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import phys
Egrid=np.exp(np.linspace(np.log(0.001),np.log(2.0),3000))
P=phys.mu_parts(Egrid)
def mu_fun_factory():
    def f(E):
        lE=np.log(E)
        lg=np.log(Egrid)
        out={}
        # interpolate log-values where positive, linear otherwise
        for k in ['tot','C','trC','coh','pe','pair']:
            out[k]=np.interp(E,Egrid,P[k])
        return out
    return f
mu_fun=mu_fun_factory()
'''
open('/tmp/work/mut.py','w').write(tab)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, mc as MC, mut, phys, numpy as np, time
# check interpolation accuracy
Et=np.array([0.02,0.1,0.3,1.1732,1.3325])
a=mut.mu_fun(Et); b=phys.mu_parts(Et)
print({k:np.round(a[k]/b[k],4) for k in ['tot','C','trC']})
t0=time.time()
T,kerma,flu,ev=MC.run(20000,2,mut.mu_fun,[105.0,100.5],np.arange(0,1.4001,0.05),np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0]))
print(time.time()-t0, T['Lp']/20000/MC.VOL*MC.F0, T['Ls']/20000/MC.VOL*MC.F0)
