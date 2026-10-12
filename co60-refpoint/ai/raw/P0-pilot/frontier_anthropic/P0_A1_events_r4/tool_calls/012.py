
import pickle, json, numpy as np, glob
pool=[]
for f in sorted(glob.glob('/tmp/work/events_1[1-4].pkl')):
    pool+=pickle.load(open(f,'rb'))
print(len(pool))
rng=np.random.default_rng(2024)
idx=rng.choice(len(pool),200,replace=False)
sample=[pool[i] for i in idx]
from collections import Counter
print(Counter(e[0] for e in sample))
events=[]
for (pr,ein,eout,cs,te,ce,tp) in sample:
    events.append({"process":pr,"e_in_MeV":round(ein,5),
                   "e_out_MeV":None if eout is None else round(eout,5),
                   "cos_scatter":None if cs is None else round(cs,6),
                   "t_e_MeV":round(te,5),
                   "cos_electron":None if ce is None else round(ce,6),
                   "t_positron_MeV":None if tp is None else round(tp,5)})
method=("Analog Monte Carlo (own numpy code, no transport package). Point source at 100 cm SSD, directions sampled uniformly in solid angle "
        "within the 10x10 cm^2 pyramid; 2.0 MeV photons transported through the 30 cm water cube (vacuum outside) with exponential free paths "
        "(total mu/rho from Klein-Nishina incoherent with ~2% binding reduction, Rayleigh, photoelectric and pair/triplet-free pair estimates). "
        "Compton: Klein-Nishina rejection sampling, no binding/scattering function; Rayleigh: small-angle Gaussian; pair: energy split uniform, "
        "two 511 keV annihilation photons isotropic; photoelectric angle from Sauter-like law. All interactions in the 1 cm radius, 0.2 cm slab at 10 cm depth "
        "were recorded (~1.2e8 histories, ~44,000 events); 200 were drawn at random without replacement.")
print(len(method.split()))
out={"method":method,"events":events}
s=json.dumps(out,indent=1)
print(s[:800])
