
import sys, pickle, time; sys.path.insert(0,'/tmp/work')
import numpy as np, elec
tot=pickle.load(open('/tmp/work/tot.pkl','rb'))
B=tot['births']; N=tot['N']; VCYL=np.pi*0.2; PHI=1/(0.0099751*110**2)
Ball=B[:,3].sum()/(N*VCYL*PHI)
t=time.time(); res=elec.run_e(B,Ball,100000,7); print(time.time()-t, res['steps'])
pickle.dump(res,open('/tmp/work/eres1.pkl','wb'))
print(res['fl_tot'], res['fl_del']/res['fl_tot'], res['rad']/res['col'], res['dep_tot'], res['T0s']/100000*Ball)
