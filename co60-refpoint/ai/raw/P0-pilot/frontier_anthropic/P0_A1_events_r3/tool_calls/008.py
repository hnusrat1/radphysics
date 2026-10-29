
import pickle, json, collections
ev=pickle.load(open('events_6000000.pkl','rb'))
print(len(ev), collections.Counter(e[0] for e in ev))
print(collections.Counter((e[0], e[1]==2.0) for e in ev))
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
method=("Analog Monte Carlo (numpy). Source photons sampled uniformly in solid angle in the 10x10 cm2 pyramid at SSD 100 cm; "
 "straight-line transport in homogeneous water (30 cm cube, vacuum outside), free paths sampled with total coefficient "
 "(Klein-Nishina Compton, rejection-sampled exactly; Rayleigh with (1+cos^2)/2 angular shape; pair ~0.48 mb-free estimate 4.8e-4 cm2/g at 2 MeV; photoelectric approximated as 2.76e-3(0.1/E)^3 cm2/g; "
 "coherent approximated). Pair: annihilation photons 2x0.511 MeV tracked. Each interaction whose point lies in the 1 cm radius, 0.2 cm thick cylinder at 10 cm depth was recorded; "
 "6x10^6 primary histories gave 2283 events; 200 drawn at random without replacement. Electron energies/angles from kinematics (Compton); pair energy split uniform, angles approximate; binding ignored.")
out={"method":method,"events":events}
print(len(events))
print(json.dumps(out)[:600])
