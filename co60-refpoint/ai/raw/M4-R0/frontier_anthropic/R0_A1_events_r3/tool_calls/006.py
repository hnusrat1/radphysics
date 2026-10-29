
import numpy as np, collections, random as pyr
ev=[]
for s in range(1,5):
    ev+=json.load(open(f"run{s}.json"))
W=np.array([e["w"] for e in ev]); print(len(ev), W.sum(), W.min(), W.max())
procs_=collections.Counter(e["process"] for e in ev)
print({k:(v, round(sum(e["w"] for e in ev if e["process"]==k)/W.sum(),5)) for k,v in procs_.items()})
rng=np.random.default_rng(2024)
idx=rng.choice(len(ev), size=200, replace=True, p=W/W.sum())
print("unique", len(set(idx)))
sample=[{k:v for k,v in ev[i].items() if k!="w"} for i in idx]
c=collections.Counter(e["process"] for e in sample); print(c)
print(collections.Counter(e["e_in_MeV"] for e in sample))
