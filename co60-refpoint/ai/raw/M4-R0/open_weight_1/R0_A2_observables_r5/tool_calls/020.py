code = open('/tmp/photonmc.py').read()
old = """            mu_i = mu[ii]
            rp = rng.random(len(ii))*mu_i
            muc = mut_at(E_int, mu_coh); mui = mut_at(E_int, mu_inc); mup = mut_at(E_int, mu_pe); mupair = mut_at(E_int, mu_pair)
            c1=muc; c2=c1+mui; c3=c2+mup; c4=c3+mupair
            proc = np.where(rp<c1, 0, np.where(rp<c2, 1, np.where(rp<c3, 2, 3)))
            proc = np.where(rp>=c4, 3, proc)"""
new = """            muc = mut_at(E_int, mu_coh); mui = mut_at(E_int, mu_inc); mup = mut_at(E_int, mu_pe); mupair = mut_at(E_int, mu_pair)
            c1=muc; c2=c1+mui; c3=c2+mup; c4=c3+mupair
            mu_own = c4
            rp = rng.random(len(ii))*mu_own
            proc = np.where(rp<c1, 0, np.where(rp<c2, 1, np.where(rp<c3, 2, 3)))
            proc = np.where(rp>=c4, 3, proc)"""
assert old in code
code=code.replace(old,new)
# guard pair Tpair
old2 = "                Tpair = E_int[m_pair]-1.022"
new2 = "                Tpair = np.maximum(E_int[m_pair]-1.022,0.0)"
assert old2 in code
code=code.replace(old2,new2)
open('/tmp/photonmc.py','w').write(code)
import importlib, sys, time
sys.path.insert(0,'/tmp')
import photonmc
importlib.reload(photonmc)
photonmc.PAIRDBG0 = None
t0=time.time()
res = photonmc.photon_mc(2000000, seed=11, verbose=False, debug=False)
print('elapsed', time.time()-t0)
print('int_all', res['int_all'], 'etr_all', res['etr_all'])