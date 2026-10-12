
rng=np.random.default_rng(12345)
idx=rng.choice(len(ev),200,replace=False)
sub=[ev[i] for i in idx]
def r6(x): return None if x is None else float(f"{x:.6g}")
out=[]
for e in sub:
    d={"process":e['process'],"e_in_MeV":r6(e['e_in_MeV']),"e_out_MeV":r6(e['e_out_MeV']),"cos_scatter":r6(e['cos_scatter']),
       "t_e_MeV":r6(e['t_e_MeV']),"cos_electron":r6(e['cos_electron']),"t_positron_MeV":r6(e['t_positron_MeV'])}
    out.append(d)
print(collections.Counter(d['process'] for d in out), sum(d['e_in_MeV']==2.0 for d in out))
method=("Own Monte Carlo (no transport code available): 10^7 source photons from a point source at 100 cm SSD, "
 "directions sampled uniformly in solid angle within the 10x10 cm2 pyramid; photons tracked through the 30 cm water cube "
 "(rho 1, vacuum outside) with exponential free paths and water coefficients: Klein-Nishina Compton (free-electron, no binding), "
 "approximate photoelectric (~E^-3), coherent (~E^-1.2, small-angle form-factor model), pair production (interpolated Bethe-Heitler-like). "
 "Every interaction whose point lies in the 1 cm radius, 0.2 cm thick cylinder at 10 cm depth was recorded (3708 events in pool); 200 were drawn uniformly without replacement. "
 "Compton KN sampling, pair energy shared uniformly, annihilation photons tracked as secondaries; positron range and bremsstrahlung neglected.")
print(len(method.split()))
json.dump({"method":method,"events":out},open('answer.json','w'))
print(json.dumps({"method":method,"events":out[:3]})[:800])
