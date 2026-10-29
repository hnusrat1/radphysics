
xs_code = xs_code.replace("def fbind(E): return 1-0.046*np.clip(E/2.0,1e-3,None)**0.42","def fbind(E): return 0.98+0*E")
xs_code = xs_code.replace("return 2.66e-3*f**1.5","return 1.45e-3*f**1.5")
with open('/tmp/work/xs.py','w') as f: f.write(xs_code)
import importlib; importlib.reload(xs)
for E in [2.0,1.5,1.0,0.5,0.2,0.1,0.05]:
    print(E, xs.mu_tot(E), xs.mu_inc(E), xs.mu_pair(E), xs.mu_pe(E))
