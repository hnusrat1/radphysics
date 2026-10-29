
import os, sys
print(os.getcwd(), os.listdir('.')[:20])
sys.path.insert(0, os.getcwd())
open('xs.py','w').write(code)
import xs
print(xs.mu_tot(2.0), xs.mu_inc(2.0), xs.mu_tot(0.1), xs.mu_tot(0.5), xs.mu_tot(1.0))
