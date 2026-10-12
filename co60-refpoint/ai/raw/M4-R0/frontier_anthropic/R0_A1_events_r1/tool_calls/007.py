
import random
rs=random.Random(2024)
samp=rs.sample(tot,200)
events=[]
for e in samp:
    proc,ein,eout,cs,te,ce,tp=e
    events.append({"process":proc,"e_in_MeV":round(ein,5),
                   "e_out_MeV":None if eout is None else round(eout,5),
                   "cos_scatter":None if cs is None else round(cs,5),
                   "t_e_MeV":round(te,5),
                   "cos_electron":None if ce is None else round(ce,5),
                   "t_positron_MeV":None if tp is None else round(tp,5)})
print(collections.Counter(x['process'] for x in events), collections.Counter(x['e_in_MeV'] for x in events))
out={"method":"Analog Monte Carlo (numpy) of point source 100 cm SSD, 10x10 field, uniform solid-angle pyramid sampling, two Co-60-like lines (1.1732/1.3325 MeV, equal weight), 30 cm water cube in vacuum. Photons transported with exponential free paths and water cross sections (Klein-Nishina Compton with electron-density scaling, Rayleigh, photoelectric and pair approximations; annihilation photons tracked). Only interactions inside the 1 cm radius, 0.2 cm thick cylinder at 5 cm depth were recorded (~21,700 from 4e7 histories); 200 were drawn uniformly at random without replacement. Bremsstrahlung and fluorescence neglected; pair energy split uniform; electron angles approximate.",
     "events":events}
json.dump(out,open('answer.json','w'))
print(len(events)); print(json.dumps(events[:3]))
