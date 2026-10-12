
import numpy as np, time
rng=np.random.default_rng(12345)
EC=0.02  # cutoff MeV
def mu_parts(E):
    sKN=ne*sigma_KN(E)
    coh=sKN*0.032*(E/0.1)**(-1.2)
    pe=0.0028*(E/0.1)**(-3.0)
    pair=np.where(E>1.022, 0.0005*np.clip((E-1.022)/0.978,0,None)**2, 0.0)
    return sKN,coh,pe,pair
def mu_tot(E):
    a,b,c,d=mu_parts(E); return a+b+c+d

# Kahn/rejection KN sampling of cos theta
def sample_KN(E):
    k=E/me
    n=len(E)
    out_c=np.empty(n)
    todo=np.arange(n)
    # envelope: f(mu)=eps^2(eps+1/eps-(1-mu^2)), eps=1/(1+k(1-mu)); fmax computed on grid per unique energy (use global max bound)
    mus=np.linspace(-1,1,401)
    while todo.size>0:
        kk=k[todo]
        m=rng.uniform(-1,1,todo.size)
        eps=1/(1+kk*(1-m))
        f=eps**2*(eps+1/eps-(1-m*m))
        fmax=np.maximum(2.0, 2.0)  # f <= 2*... bound check below
        acc=rng.uniform(0,1,todo.size)*4.0 < f   # envelope 4 >= f max
        out_c[todo[acc]]=m[acc]
        todo=todo[~acc]
    return out_c
# verify envelope bound
mm=np.linspace(-1,1,2001)
for E in [0.02,0.1,0.5,1.1732,1.3325]:
    k=E/me; eps=1/(1+k*(1-mm)); f=eps**2*(eps+1/eps-(1-mm**2)); print(E,f.max())
