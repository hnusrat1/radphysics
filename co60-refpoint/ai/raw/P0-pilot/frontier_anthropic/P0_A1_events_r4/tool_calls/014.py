
import pickle, json, numpy as np, glob
from collections import Counter
pool=[]
for f in sorted(glob.glob('/tmp/work/events_1[1-4].pkl')):
    pool+=pickle.load(open(f,'rb'))
rng=np.random.default_rng(2024)
idx=rng.choice(len(pool),200,replace=False)
sample=[pool[i] for i in idx]
events=[]
for (pr,ein,eout,cs,te,ce,tp) in sample:
    events.append({"process":pr,"e_in_MeV":round(ein,5),
                   "e_out_MeV":None if eout is None else round(eout,5),
                   "cos_scatter":None if cs is None else round(cs,6),
                   "t_e_MeV":round(te,5),
                   "cos_electron":None if ce is None else round(ce,6),
                   "t_positron_MeV":None if tp is None else round(tp,5)})
method=("Analog Monte Carlo written in numpy (no transport package). Point source 100 cm from surface; directions sampled uniformly in solid angle "
        "inside the 10x10 cm^2 pyramid. 2 MeV photons traced through the 30 cm water cube with exponential free paths and water cross sections "
        "(Klein-Nishina Compton, Rayleigh, photoelectric, pair). Compton by Klein-Nishina rejection; annihilation photons tracked. "
        "All interactions inside the 1 cm radius, 0.2 cm slab at 10 cm depth were recorded from ~1.2e8 histories (~44,600 events); 200 drawn at random without replacement.")
out={"method":method,"events":events}
with open('/tmp/work/answer.json','w') as f: json.dump(out,f)
print(len(pool), Counter(e["process"] for e in events), len(events))
print(json.dumps(out))
