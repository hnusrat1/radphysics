
pdd = r'''
import sys, time, gc
sys.path.insert(0,'/tmp/work')
import numpy as np, phys, mc
depths=[0.25,0.5,0.75,1.0,5.0]
acc=[]
t=time.time()
for s in range(6):
    r=mc.run(400000,depths,np.random.default_rng(900+s),store_for=5.0)
    acc.append(r['dscat']/400000*phys.OMEGA*105**2)
    gc.collect()
acc=np.array(acc)
mu=phys.mu_tot(phys.Lines)
def prim(z):
    return sum(0.5*phys.ETR(E)*np.exp(-m*z)*(105/(100+z))**2 for E,m in zip(phys.Lines,mu))
P=np.array([prim(z) for z in depths])
D=P[None,:]+acc
for i,z in enumerate(depths):
    print(z, 'prim',P[i],'scat', acc[:,i].mean(), 'sem', acc[:,i].std(ddof=1)/np.sqrt(len(acc)), 'D', D[:,i].mean())
pdd=100*D[:,4]/D[:,1]
print('PDD5 with dmax=0.5:', pdd.mean(), pdd.std(ddof=1)/np.sqrt(len(pdd)))
pdd2=100*D[:,4]/D[:,0]
print('PDD5 vs 0.25:', pdd2.mean())
print('time', time.time()-t)
'''
open('/tmp/work/pdd.py','w').write(pdd)
import subprocess
r=subprocess.run(['python3','/tmp/work/pdd.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout, r.stderr[-2000:])
