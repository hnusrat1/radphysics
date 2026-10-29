
import sys; sys.path.insert(0,'/tmp/work')
import ana
for E in [0.1,0.5,1.1732,1.3325]:
    s,me=ana.kn_eps_stats(E); print(E, s/ana.kn_total(E), 1-me)
