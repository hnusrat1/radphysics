import pandas as pd, numpy as np
arr=[]
for l in open('/tmp/sample.txt'):
 a=l.split(); arr.append([int(a[0])]+[float(x) for x in a[1:]])
a=np.array(arr); print(np.bincount(a[:,0].astype(int))); print('E ranges',a[:,1].min(),a[:,1].max(),np.percentile(a[:,1],[0,10,50,90,100])); print('scat mu',np.percentile(a[a[:,0]==0,3],[0,10,50,90,100])); print('max event energies low')