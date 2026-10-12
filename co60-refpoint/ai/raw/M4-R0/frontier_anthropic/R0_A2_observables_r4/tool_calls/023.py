
src=open('/tmp/work/ana2.py').read()
src=src.replace("pr_trE=np.array([phi[i]*(mu_inc(LN[i])*LN[i]*ftr_f(LN[i])), phi[i]*mu_pe(LN[i])*LN[i], phi[i]*mu_pair(LN[i])*max(LN[i]-1.022,0), 0.0] for i in range(2)).sum(0)",
"pr_trE=np.array([[phi[i]*(mu_inc(LN[i])*LN[i]*ftr_f(LN[i])), phi[i]*mu_pe(LN[i])*LN[i], phi[i]*mu_pair(LN[i])*max(LN[i]-1.022,0), 0.0] for i in range(2)]).sum(0)")
open('/tmp/work/ana2.py','w').write(src)
import sys,time; sys.path.insert(0,'/tmp/work')
t=time.time()
import ana2
print('time',time.time()-t)
