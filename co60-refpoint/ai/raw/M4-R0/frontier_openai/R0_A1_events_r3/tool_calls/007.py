import numpy as np
a=np.array([[int(x[0])]+[float(y) for y in x[1:]] for x in map(str.split,open('/tmp/sample.txt'))]); print(np.bincount(a[:,0].astype(int))); print('E ranges',a[:,1].min(),a[:,1].max(),np.percentile(a[:,1],[0,10,50,90,100])); print('proc E quantiles');
for p in range(4):
 q=a[a[:,0]==p,1]; print(p,len(q), np.percentile(q,[0,25,50,75,100]) if len(q) else '-')
