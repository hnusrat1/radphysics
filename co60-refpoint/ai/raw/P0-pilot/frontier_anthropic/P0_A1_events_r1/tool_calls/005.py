
code=open('mc.py').read()
old=code[code.index('def sample_compton(E):'):code.index('def iso_dir(n):')]
new='''def sample_compton(E):
    # proposal density in eps: 1/(eps*ln(1/epsmin)); f/(1/eps) = 1+eps^2-eps*sin^2 <= 2
    n=E.size; k=E/me
    epsmin=1/(1+2*k)
    L=np.log(1/epsmin)
    out_eps=np.empty(n)
    todo=np.arange(n)
    while todo.size:
        kk=k[todo]; L_=L[todo]; em=epsmin[todo]
        eps=np.exp(-L_*rng.random(todo.size))    # in [epsmin,1]
        c=1-(1/eps-1)/kk
        ratio=(1+eps*eps-eps*(1-c*c))/2.0
        acc=rng.random(todo.size) < ratio
        out_eps[todo[acc]]=eps[acc]
        todo=todo[~acc]
    return out_eps
'''
code=code.replace(old,new)
# remove the faulty assert block
i0=code.index('# check the bound numerically'); i1=code.index('def iso_dir(n):')
code=code[:i0]+code[i1:]
open('mc.py','w').write(code)
import subprocess,time
t=time.time()
r=subprocess.run(['python','mc.py','1','20000'],capture_output=True,text=True)
print(r.stdout[-2000:], r.stderr[-3000:], time.time()-t)
