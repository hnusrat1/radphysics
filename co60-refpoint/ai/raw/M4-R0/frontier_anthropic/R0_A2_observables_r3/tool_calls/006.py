
import sys, os
sys.path.insert(0,'/tmp/work')
code=open('/tmp/work/phys.py').read().replace('np.trapz','np.trapezoid')
open('/tmp/work/phys.py','w').write(code)
import importlib
import phys
E=np.array([1.1732,1.3325])
print(phys.mu_tot(E), phys.mu_pair(E), phys.mu_coh(E)/phys.mu_tot(E))
print(phys.eps_mean(E))
print(phys.mu_inc(E)*(1-phys.eps_mean(E)) + phys.mu_pair(E)*(1-1.022/E)+ phys.mu_pe(E))
