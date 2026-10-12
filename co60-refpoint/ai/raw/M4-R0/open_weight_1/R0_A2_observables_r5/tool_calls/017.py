# quick debug: check pair events in scoring volume
code = open('/tmp/photonmc.py').read()
# insert debug print after etr_all[3] update for pair
old = "                    etr_all[3]+= Tpair[ssel].sum()"
new = "                    etr_all[3]+= Tpair[ssel].sum()\n                    if globals().get('PAIRDBG') and len(Tpair[ssel]):\n                        print('PAIRDBG E', E_int[m_pair][ssel], 'Tpair', Tpair[ssel], 'ic', ic_int[m_pair][ssel])"
assert old in code
code=code.replace(old,new)
open('/tmp/photonmc_dbg.py','w').write(code)
import importlib, sys, time
sys.path.insert(0,'/tmp')
import photonmc_dbg, numpy as np
importlib.reload(photonmc_dbg)
photonmc_dbg.PAIRDBG=True
res = photonmc_dbg.photon_mc(1000000, seed=7, verbose=False, debug=False)
print('int_all', res['int_all'])
print('etr_all', res['etr_all'])