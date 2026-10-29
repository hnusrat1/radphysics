# Fix cyl_seg_length: when disc<=0, no intersection -> set tinc>toutc
code = open('/tmp/photonmc.py').read()
old = """    disc=b*b-4*a*c
    ok=disc>0
    sq=np.sqrt(np.where(ok,disc,0.0))
    t1c=(-b-sq)/(2*a); t2c=(-b+sq)/(2*a)
    tinc=np.maximum(np.where(ok,t1c,0.0),0.0)
    toutc=np.minimum(np.where(ok,t2c,s),s)"""
new = """    disc=b*b-4*a*c
    ok=disc>0
    sq=np.sqrt(np.where(ok,disc,0.0))
    t1c=np.where(ok, (-b-sq)/(2*a), 1.0)
    t2c=np.where(ok, (-b+sq)/(2*a), 0.0)
    tinc=np.maximum(t1c,0.0)
    toutc=np.minimum(t2c,s)"""
assert old in code
code = code.replace(old,new)
open('/tmp/photonmc.py','w').write(code)
import importlib, sys, time
sys.path.insert(0,'/tmp')
import photonmc
importlib.reload(photonmc)
t0=time.time()
res = photonmc.photon_mc(50000, seed=42, verbose=False, debug=False)
print('time', time.time()-t0)
print('primL', res['fl_prim'].sum(), 'scatL', res['fl_scat'].sum())
print('n_cyl_seg prim/scat', res['n_cyl_seg_prim'], res['n_cyl_seg_scat'])
print('int_all', res['int_all'])
print('src', len(res['src']['T']), 'L3', len(res['L3_Te']))