
events=[]
for r in rows:
    p=r['process']
    ev={'process':p,'e_in_MeV':r['e_in'],'e_out_MeV':r['e_out'],'cos_scatter':r['cos_s'],
        't_e_MeV':r['te'],'cos_electron':r['ce'],'t_positron_MeV':r['tp']}
    if p=='compton':
        assert abs(r['e_in']-r['e_out']-r['te'])<2e-6 and r['tp'] is None
        assert r['ce'] is not None and r['cos_s'] is not None
    if p=='rayleigh':
        ev['e_out_MeV']=r['e_in']; ev['t_e_MeV']=0.0; ev['cos_electron']=None
    events.append(ev)
# check rayleigh row
print([e for e in events if e['process']!='compton'])
print(min(e['t_e_MeV'] for e in events), max(e['e_in_MeV'] for e in events))
out={'method':"Analog photon transport in the water cube (Klein-Nishina Compton with incoherent-only angular sampling, approximate Rayleigh, photoelectric and pair cross sections, 1.1732/1.3325 MeV sources sampled uniformly in solid angle over the 10x10 cm pyramid, box geometry with vacuum outside). Every photon collision (including annihilation photons) gave a next-event estimate of the probability that its next interaction falls in the scoring cylinder, with the process and outgoing state sampled; events were drawn from the weighted pool of 2e7 source photons by weighted reservoir sampling (200 kept), so the sample is proportional to the true interaction rate in the volume. Bremsstrahlung and fluorescence neglected.",
     'events':events}
print(len(events))
s=json.dumps(out,separators=(',',':'))
print(len(s))
open('answer.json','w').write(s)
