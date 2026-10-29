import numpy as np, collections
arr=np.loadtxt('/tmp/e.csv')
print(arr.shape,collections.Counter(arr[:,0].astype(int)))
for p in range(4):
 a=arr[arr[:,0]==p]
 print(p, len(a), np.quantile(a[:,1],[0,.1,.25,.5,.75,.9,1]) if len(a) else '')
print('primary close2',np.mean(np.abs(arr[:,1]-2)<1e-6))