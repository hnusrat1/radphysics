
code2=open('mc.py').read()
code2=code2.replace('''def sample_KN(E,n):
    k=E/me
    out=np.empty(n); todo=np.arange(n)
    mg=np.linspace(-1,1,20001); e=1/(1+k*(1-mg)); fmax=(e**2*(e+1/e-(1-mg**2))).max()*1.02
    while todo.size:
        m=rng.uniform(-1,1,todo.size); e=1/(1+k*(1-m))
        f=e**2*(e+1/e-(1-m**2))
        acc=rng.uniform(0,fmax,todo.size)<f
        out[todo[acc]]=m[acc]; todo=todo[~acc]
    return out''','''def sample_KN(E,n):
    # E: array of energies (n,), vectorized rejection with envelope f<=2*(1+k)... use bound 2*(1+2k)/(1+2k)**? numerically safe bound
    k=np.asarray(E)/me*np.ones(n)
    out=np.empty(n); todo=np.arange(n)
    # envelope: f(mu) <= max over mu of eps^2(eps+1/eps-sin^2); eps<=1 so f<=2*1 ; use fmax per energy: at mu=1 f=2 (eps=1). f max = 2 for all k? check below
    while todo.size:
        kk=k[todo]
        m=rng.uniform(-1,1,todo.size); e=1/(1+kk*(1-m))
        f=e**2*(e+1/e-(1-m**2))
        acc=rng.uniform(0,2.0,todo.size)<f
        out[todo[acc]]=m[acc]; todo=todo[~acc]
    return out''')
code2=code2.replace('''        mu=np.empty(idx.size)
        for E_val in np.unique(Ei):
            sel=Ei==E_val
            mu[sel]=sample_KN(E_val,sel.sum())''','''        mu=sample_KN(Ei,idx.size)''')
open('mc.py','w').write(code2)
# check envelope f<=2 numerically
import numpy as np
me=0.51099895
for k in [0.2/me,1/me,2/me,3.9139,5]:
    m=np.linspace(-1,1,100001); e=1/(1+k*(1-m)); f=e**2*(e+1/e-(1-m**2)); print(k,f.max())
