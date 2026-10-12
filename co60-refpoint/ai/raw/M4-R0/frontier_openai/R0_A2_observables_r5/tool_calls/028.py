import numpy as np
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
# interpolate each coefficient linearly loglog with fallback linear for zero

def coeff(E, col):
    k=np.searchsorted(x[:,0],E); k=np.clip(k,1,len(x)-1); out=[]
    for e,idx in zip(np.atleast_1d(E),np.atleast_1d(k)):
        a,b=x[idx-1,col],x[idx,col]; e0,e1=x[idx-1,0],x[idx,0]
        if a>0 and b>0: v=np.exp(np.log(a)+(np.log(e/e0)/np.log(e1/e0))*np.log(b/a))
        else: v=a+(b-a)*(e-e0)/(e1-e0)
        out.append(v)
    return np.array(out)
phspec=np.zeros(28);phspec[23]=.3607301346;phspec[26]=.368286654
sh=np.array([0,1.896757073e-06,2.711916391e-05,8.22782131e-05,0.000713897557,0.001121546131,0.002082667525,0.002368731648,0.001277173286,0.0007390768456,0.0005806524,0.0004283673863,0.0003738276051,0.0003669613753,0.000337788361,0.0003215975074,0.0003226709403,0.0002979864319,0.0002858250798,0.000261426193,0.0002581786833,0.0002705731785,0.0002730533806,0.1347732991,0.0001287440712,0.0001211507842,0.128187807,0])
Ebin=np.arange(28)*.05+.025
# indices co/incoh/photo/pair sum separate
for name,spec,energies in [('p',phspec,None),('s',sh,Ebin)]:
    E=np.array([1.1732,1.3325]) if name=='p' else energies
    W=np.array([spec[23],spec[26]]) if name=='p' else spec
    p1=coeff(E,1)*W;p2=coeff(E,2)*W;p3=coeff(E,3)*W;p4=(coeff(E,4)+coeff(E,5))*W
    vals=np.array([p2.sum(),p3.sum(),p4.sum(),p1.sum()]);print(name,vals/vals.sum(),vals.sum())
