
import numpy as np, json, subprocess
d=np.load('sample_99.npz')
names=['compton','rayleigh','photoelectric','pair']
events=[]
for i in range(len(d['proc'])):
    p=names[int(d['proc'][i])]
    eo=float(d['Eout'][i]); cs=float(d['cosS'][i]); te=float(d['te'][i]); ce=float(d['cose'][i]); tp=float(d['tpos'][i])
    e_in=float(d['Ein'][i])
    ev={'process':p,'e_in_MeV':round(e_in,6),
        'e_out_MeV':None if np.isnan(eo) else round(eo,6),
        'cos_scatter':None if np.isnan(cs) else round(cs,5),
        't_e_MeV':round(te,6),
        'cos_electron':None if np.isnan(ce) else round(ce,5),
        't_positron_MeV':None if np.isnan(tp) else round(tp,6)}
    if p=='rayleigh':
        ev['e_out_MeV']=round(e_in,6); ev['t_e_MeV']=0.0; ev['cos_electron']=None
    events.append(ev)
out={'method':"Analog photon transport in the 30 cm water cube (vacuum outside) with Klein-Nishina Compton sampling and approximate Rayleigh, photoelectric and pair cross sections; annihilation photons followed. Sources 1.1732/1.3325 MeV, equal weight, directions uniform in solid angle over the 10x10 cm pyramid. At every collision, next-event estimation gives the probability that the photon's next interaction lies in the 1 cm radius, 0.2 cm slab at 5 cm depth; process, outgoing photon and electron are sampled from that conditional distribution. 200 events were drawn from the weighted pool of about 2e7 source photons (weighted reservoir sampling, weight proportional to interaction rate). Bremsstrahlung and fluorescence neglected.",
     'events':events}
s=json.dumps(out,separators=(',',':'))
print(len(events), len(s))
print(s)
