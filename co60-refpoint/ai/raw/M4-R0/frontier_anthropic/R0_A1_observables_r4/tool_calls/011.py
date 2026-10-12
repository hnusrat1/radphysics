
import sys; sys.path.insert(0,'/tmp/work')
import mc as MC, mut, numpy as np, time, pickle
cos_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
Eph=np.arange(0,1.4001,0.05)
t0=time.time()
res=[]
for s in range(4):
    res.append(MC.run(250000,100+s,mut.mu_fun,[105.0,100.5],Eph,cos_edges))
print('time',time.time()-t0)
pickle.dump(res,open('/tmp/work/res1.pkl','wb'))
