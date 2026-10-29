
import json,collections,numpy as np
ev=[]
for s in range(2,6): ev+=json.load(open(f'events_{s}.json'))
rng=np.random.default_rng(12345)
idx=rng.choice(len(ev),200,replace=False)
sub=[ev[i] for i in idx]
def r6(x): return None if x is None else float(f"{x:.6g}")
out=[]
for e in sub:
    out.append({"process":e['process'],"e_in_MeV":r6(e['e_in_MeV']),"e_out_MeV":r6(e['e_out_MeV']),"cos_scatter":r6(e['cos_scatter']),
       "t_e_MeV":r6(e['t_e_MeV']),"cos_electron":r6(e['cos_electron']),"t_positron_MeV":r6(e['t_positron_MeV'])})
print(collections.Counter(d['process'] for d in out), sum(d['e_in_MeV']==2.0 for d in out))
method=("Own Monte Carlo (no transport code available): 10^7 source photons from a point source at 100 cm SSD, "
 "directions sampled uniformly in solid angle within the 10x10 cm2 pyramid; photons tracked through the 30 cm water cube "
 "(rho 1, vacuum outside) with exponential free paths and approximate water coefficients: Klein-Nishina Compton (free electron, no binding), "
 "approximate photoelectric (~E^-3), coherent (~E^-1.2, crude form-factor angles), pair production (interpolated). "
 "Every interaction whose point lies in the 1 cm radius, 0.2 cm thick cylinder at 10 cm depth was recorded (3708-event pool); 200 drawn uniformly without replacement. "
 "Pair energy shared uniformly; annihilation photons tracked as secondaries at the pair point; positron range and bremsstrahlung neglected.")
print(len(method.split()))
json.dump({"method":method,"events":out},open('answer.json','w'))
print(json.dumps({"method":method,"events":out}))
