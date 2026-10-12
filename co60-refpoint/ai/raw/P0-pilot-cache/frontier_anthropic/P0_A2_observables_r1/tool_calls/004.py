
code = r'''
import numpy as np
xc=np.loadtxt('/tmp/data/xcom_water.txt'); es=np.loadtxt('/tmp/data/estar_water.txt')
ME=0.51099895
E_x=xc[:,0]
TAB={'coh':xc[:,1],'inc':xc[:,2],'pe':xc[:,3],'pr':xc[:,4]+xc[:,5]}
def mu(E,which):
    E=np.atleast_1d(np.asarray(E,float))
    col=TAB[which]
    out=np.zeros_like(E)
    m=(E>=0.001)&(E<=20)
    if which=='pr': m=m&(E>1.022)
    lc=np.log(np.maximum(col,1e-30))
    if which=='pr':
        # pair table has zeros below threshold; use log interp on positive entries
        pos=col>0
        out[m]=np.exp(np.interp(np.log(E[m]),np.log(E_x[pos]),lc[pos]))
    else:
        out[m]=np.exp(np.interp(np.log(E[m]),np.log(E_x),lc))
    return out
def mut(E): return mu(E,'coh')+mu(E,'inc')+mu(E,'pe')+mu(E,'pr')
# ESTAR
ET=es[:,0]; SCOL=es[:,1]; SRAD=es[:,2]; STOT=es[:,3]; RANGE=es[:,4]; YLD=es[:,5]
def interpE(T,arr,lin=False):
    T=np.atleast_1d(np.asarray(T,float))
    return np.exp(np.interp(np.log(np.clip(T,ET[0],ET[-1])),np.log(ET),np.log(np.maximum(arr,1e-30))))
def Scol(T): return interpE(T,SCOL)
def Srad(T): return interpE(T,np.maximum(SRAD,1e-10))
def Stot(T): return interpE(T,STOT)
def Yld(T): return np.interp(np.log(np.clip(T,ET[0],ET[-1])),np.log(ET),YLD)
'''
open('/tmp/work/xs.py','w').write(code) if __import__('os').path.isdir('/tmp/work') else None
import os; os.makedirs('/tmp/work',exist_ok=True); open('/tmp/work/xs.py','w').write(code)
import sys; sys.path.insert(0,'/tmp/work')
import xs, numpy as np
print(xs.mu(2.0,'coh'),xs.mu(2.0,'inc'),xs.mu(2.0,'pe'),xs.mu(2.0,'pr'),xs.mut(2.0))
print(xs.mut(0.5),xs.mu(1.0,'pr'), xs.Scol(1.0), xs.Yld(1.0), xs.Yld(0.01), xs.Stot(0.0101))
print(xs.es[:3,5], xs.es[-3:,0])
