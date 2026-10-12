
import sys, importlib, time, pickle
sys.path.insert(0,'/tmp/work')
import numpy as np
import sim
t=time.time()
chunks=[]
for i in range(20):
    chunks.append(sim.run_chunk(200000,100+i))
print("time",time.time()-t)
# combine
def comb(key):
    return np.mean([c[key] for c in chunks],axis=0)
tot={}
for key in ['Kd','Fd','mu_sc','tr_sc','col_sc','K_sc','spec','ang','EFl','Fl_sc','spec_over']:
    tot[key]=comb(key)
# combined segments
seg={k:np.concatenate([c['seg'][k] for c in chunks]) for k in ['r','d','E','l']}
nproc={k:sum(c['nproc'][k] for c in chunks) for k in chunks[0]['nproc']}
pickle.dump(dict(tot=tot,seg=seg,nproc=nproc),open('/tmp/work/photon_tallies.pkl','wb'))
print({k:(tot[k] if np.ndim(tot[k])==0 else np.round(tot[k],5)) for k in ['Fl_sc','EFl','spec_over']})
print("mu_sc",tot['mu_sc'],"tr_sc",tot['tr_sc'],"col",tot['col_sc'],"K",tot['K_sc'])
print("nseg in cyl",len(seg['E']))
print("Fl_sc stat check (per chunk std)", np.std([c['Fl_sc'] for c in chunks])/np.sqrt(20))
