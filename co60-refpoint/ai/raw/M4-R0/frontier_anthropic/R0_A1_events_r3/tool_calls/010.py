
import json, numpy as np
ev=[]
for s in range(1,5):
    ev+=json.load(open(f"run{s}.json"))
W=np.array([e["w"] for e in ev]); p=W/W.sum()
rng=np.random.default_rng(20240607)
idx=rng.choice(len(ev), size=200, replace=True, p=p)
def r(x, n=6):
    return None if x is None else float(f"{x:.{n}g}")
events=[]
for i in idx:
    e=ev[i]
    events.append({
        "process": e["process"],
        "e_in_MeV": r(e["e_in_MeV"]),
        "e_out_MeV": r(e["e_out_MeV"]),
        "cos_scatter": r(e["cos_scatter"]),
        "t_e_MeV": r(e["t_e_MeV"]),
        "cos_electron": r(e["cos_electron"]),
        "t_positron_MeV": r(e["t_positron_MeV"]),
    })
method=("Forced-detection Monte Carlo of photon transport in the 30 cm water cube. Source photons sampled uniformly in solid angle "
"within the 10x10 cm field at SSD 100 cm, energies 1.1732/1.3325 MeV with equal probability. Compton (Klein-Nishina sampling, "
"electron kinematics), approximate Rayleigh, pair production (above 1.022 MeV, annihilation photons tracked) and photoelectric "
"absorption were modelled with approximate NIST-like water coefficients; photons below 0.1 MeV absorbed. Every collision point "
"is sampled with the cylinder (r=1 cm, z=4.9-5.1 cm) forced: each photon's first interaction inside the cylinder is recorded with "
"its path probability as weight, and the photon continues with the complementary weight. The 319k weighted events were resampled "
"with replacement in proportion to weight (effective sample size ~1.8e5).")
print(len(method.split()))
out={"method":method,"events":events}
s=json.dumps(out, indent=1)
print(collections.Counter(e["process"] for e in events) if False else "")
import collections
print(collections.Counter(e["process"] for e in events), len(events), len(set(idx)))
print(collections.Counter(e["e_in_MeV"] for e in events).most_common(3))
open("answer.json","w").write(s)
print(s[:1500])
