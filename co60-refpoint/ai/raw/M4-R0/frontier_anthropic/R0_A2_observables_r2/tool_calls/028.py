
src=open('/tmp/mc.py').read()
old="            slab_add(tallies, pos[sel], d[sel], lo[sel], hi[sel], Es, fl, mtr, TLs)\n"
assert old in src
src=src.replace(old,"")
anchor="        # move\n        newpos=pos+L[:,None]*d\n"
assert anchor in src
add=("        TLz,loz,hiz=clip_cyl(pos,d,L,0.0,10.0,R)\n"
     "        selz=TLz>0\n"
     "        if selz.any():\n"
     "            Ez=E[selz]; flz=flag[selz].astype(np.int64)\n"
     "            slab_add(tallies,pos[selz],d[selz],loz[selz],hiz[selz],Ez,flz,mu_tr_coef(Ez),TLz[selz])\n")
src=src.replace(anchor, add+anchor)
open('/tmp/mc.py','w').write(src)
import sys, time, pickle
sys.path.insert(0,'/tmp')
import numpy as np
import mc as M
def new_tallies():
    return dict(fl_E=np.zeros((2,M.NEB)), fl_cos=np.zeros((2,10)), W_int=np.zeros((2,4)), W_etr=np.zeros((2,4)),
                slab_TL=np.zeros((2,M.NSLAB)), slab_K=np.zeros((2,M.NSLAB)),
                recs=dict(T=[],cos=[],w=[],flag=[],proc=[]))
t=new_tallies()
t0=time.time(); NTOT=0
for c in range(24):
    M.run(1000000,900+c,t); NTOT+=1000000
    if time.time()-t0>420: break
print('N',NTOT,'time',time.time()-t0)
rec={k:np.concatenate(v) for k,v in t['recs'].items()}
t['recs']=rec; t['N']=NTOT
pickle.dump(t,open('/tmp/tallies2.pkl','wb'))
print('slabK', t['slab_K'].sum(axis=0)[45:55])
