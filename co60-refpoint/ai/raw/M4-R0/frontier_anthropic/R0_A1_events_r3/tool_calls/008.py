
w=W/W.sum()
print("ESS", 1/np.sum(w**2))
prim=np.array([e["e_in_MeV"] in (1.1732,1.3325) for e in ev])
print("primary-energy weight fraction", W[prim].sum()/W.sum(), "count", prim.sum())
# weight-based fraction of events with e_in = source energy and not Compton-degraded
print("top weights", np.sort(W)[-5:]/W.sum())
