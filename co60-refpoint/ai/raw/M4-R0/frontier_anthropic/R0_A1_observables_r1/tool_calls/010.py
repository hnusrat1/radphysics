
import sys
sys.path.insert(0,'/tmp/work')
src=open('/tmp/work/mc.py').read()
src=src.replace("""            dose[j]+=np.sum(WC*Ep*0)  # placeholder
            kerma_scat_by_pt[j]+=np.sum(WC*(Ec-Ep)*0)  # unused
            dose_scat_by_pt[j]+=np.sum(WC*(Ec-Ep))""",
"""            dose_scat_by_pt[j]+=np.sum(WC*mu_tr_c(Ep))""")
src=src.replace("def run(N, depths, rng, store_for=5.0):","def run(N, depths, rng, store_for=5.0):\n    _=0")
open('/tmp/work/mc.py','w').write(src)
import mc
t=__import__('time').time()
res=mc.run(20000, [0.5,5.0], np.random.default_rng(2), store_for=5.0)
print(__import__('time').time()-t, res['arr']['E'].size, res['dscat'])
