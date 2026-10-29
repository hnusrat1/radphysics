
import sys
sys.path.insert(0,'/tmp/work')
import phys
print(phys.mu_tot(phys.Lines), phys.mu_tr_c(phys.Lines), phys.mu_pair(phys.Lines), phys.OMEGA)
print(phys.mu_tot(np.array([0.1,0.5,1.0])), phys.mu_tr_c(np.array([0.1,0.5,1.0])))
