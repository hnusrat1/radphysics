
code = open('/tmp/work/sim.py').read()
code = code.replace("    out={k:np.concatenate(v) if len(v) else np.array([]) for k,v in seg.items()}\n    dd={k:np.concatenate(v) if len(v) else np.array([]) for k,v in dep.items()}\n    return out,dd",
"""    out={k:np.concatenate(v) if len(v) else np.array([]) for k,v in seg.items()}
    dz={k:np.concatenate(v) if len(v) else np.array([]) for k,v in dep.items()}
    # depth histograms of energy transferred (within radius rdep), bins 0.1 cm from 0 to 30
    m=(dz['r']<rdep)
    zb=np.linspace(0,30,301)
    h_d,_=np.histogram(dz['z'][m],bins=zb,weights=dz['d'][m])
    dd={'hist_d':h_d,'zb':zb}
    return out,dd""")
open('/tmp/work/sim.py','w').write(code)
import importlib
sys.path.insert(0,'/tmp/work')
import sim; importlib.reload(sim)
t=time.time()
allseg={}; hist=np.zeros(300)
NB=8; NPER=2000000
for b in range(NB):
    seg,dd=sim.run(NPER,100+b)
    for k,v in seg.items(): allseg.setdefault(k,[]).append(v)
    hist+=dd['hist_d']
seg={k:np.concatenate(v) for k,v in allseg.items()}
NTOT=NB*NPER
pickle.dump((seg,hist,NTOT),open('/tmp/work/tallies.pkl','wb'))
print(time.time()-t, NTOT, seg['tl'].size)
