
import sys, time, pickle
sys.path.insert(0,'.')
import mc2, numpy as np
depths=[0.25,0.5,0.75,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0]
t=time.time()
allres=[]
for sd in range(10,16):
    r=mc2.run(2000000,depths,seed=sd)
    allres.append(r)
    print(sd, round(time.time()-t,1), flush=True)
pickle.dump(allres,open('allres.pkl','wb'))
