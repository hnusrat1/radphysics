import sys
sys.path.insert(0, '/tmp/work')
import mc_physics
print("Imported mc_physics successfully!")
print("Has functions:", hasattr(mc_physics, 'mu_total'), hasattr(mc_physics, 'sample_compton_fast'), hasattr(mc_physics, 'R_csda_w'))
print("mu_total(2.0) =", mc_physics.mu_total(2.0) if hasattr(mc_physics, 'mu_total') else 'N/A')