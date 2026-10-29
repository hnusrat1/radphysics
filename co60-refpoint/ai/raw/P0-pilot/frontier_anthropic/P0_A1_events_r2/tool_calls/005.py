
import json, random, math
from collections import Counter
ev = json.load(open('events_3000000.json'))
random.seed(2024)
samp = random.sample(ev, 200)
print(Counter(e['process'] for e in samp))
print(Counter(e['e_in_MeV'] for e in samp).most_common(5))
for e in samp[:3]: print(e)
out = {"method": ("Analog Monte Carlo of photon transport in water only (no electron transport). Photons start from a point source 100 cm "
 "above the phantom, directions sampled uniformly in solid angle within the 10x10 cm field pyramid; 3e6 histories tracked through the 30 cm water cube. "
 "Free paths sampled from total attenuation (Klein-Nishina Compton, approximate pair 0.0006 cm2/g at 2 MeV, photoelectric and Rayleigh constants). "
 "Compton: Geant4-style Klein-Nishina sampling of energy/angle; electron momentum from conservation. Pair: energy shared uniformly, electron forward-peaked; "
 "positron annihilated at vertex giving two back-to-back 511 keV photons that are tracked as new photons. Every vertex inside the 1 cm radius, 0.2 cm thick cylinder "
 "is recorded (1160 events); 200 drawn uniformly at random without replacement. Bremsstrahlung, fluorescence, delta rays and positron range were neglected. "
 "Cross sections are approximate, so proportions are indicative only."),
 "events": [dict(process=e['process'], e_in_MeV=round(e['e_in_MeV'],6),
   e_out_MeV=None if e['e_out_MeV'] is None else round(e['e_out_MeV'],6),
   cos_scatter=None if e['cos_scatter'] is None else round(e['cos_scatter'],6),
   t_e_MeV=round(e['t_e_MeV'],6),
   cos_electron=None if e['cos_electron'] is None else round(e['cos_electron'],6),
   t_positron_MeV=None if e['t_positron_MeV'] is None else round(e['t_positron_MeV'],6)) for e in samp]}
print(len(out['events']))
s = json.dumps(out)
print(len(s))
open('answer.json','w').write(s)
