
import numpy as np, pickle, json, collections
ev=pickle.load(open('events_6000000.pkl','rb'))
rng2=np.random.default_rng(2024)
idx=rng2.choice(len(ev),200,replace=False)
sample=[ev[i] for i in idx]
print(collections.Counter(e[0] for e in sample))
events=[]
for (proc,ein,eout,cs,te,ce,tp) in sample:
    events.append({"process":proc,"e_in_MeV":round(ein,6),
                   "e_out_MeV":None if eout is None else round(eout,6),
                   "cos_scatter":None if cs is None else round(cs,6),
                   "t_e_MeV":round(te,6),
                   "cos_electron":None if ce is None else round(ce,6),
                   "t_positron_MeV":None if tp is None else round(tp,6)})
method=("Analog Monte Carlo, 6e6 primary photon histories. Source photons sampled uniformly in solid angle in the 10x10 cm2 pyramid (SSD 100 cm), "
 "straight-line transport in homogeneous water (30 cm cube, vacuum outside) with free paths from the total coefficient: Klein-Nishina Compton (exact rejection sampling, "
 "binding ignored), pair production (4.8e-4 cm2/g at 2 MeV, annihilation 2x0.511 MeV photons tracked), Rayleigh ((1+cos2)/2 shape, small coherent coefficient), "
 "photoelectric (approximate E^-3 law). Every interaction whose point lies in the r=1 cm, 0.2 cm thick cylinder at 10 cm depth was recorded (2283 events); 200 drawn at random without replacement. "
 "Compton electron energies/angles from kinematics; pair energy split uniform and angles approximate; photoelectron angle approximate.")
out={"method":method,"events":events}
print(json.dumps(out))
