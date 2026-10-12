
import json, numpy as np, collections
ev=[]
for s in range(1,5):
    ev+=json.load(open(f"run{s}.json"))
W=np.array([e["w"] for e in ev]); w=W/W.sum()
print("ESS", 1/np.sum(w**2), "N", len(ev))
prim=np.array([e["e_in_MeV"] in (1.1732,1.3325) for e in ev])
print("primary-energy weight fraction", W[prim].sum()/W.sum(), "count", prim.sum())
print("top weights", np.sort(w)[-5:])
