
import subprocess
t=r'''
import json
d=json.load(open('events.json'))
ev=d['events']
method=("Analog photon Monte Carlo of the water cube with next-event estimation to the 1 cm x 0.2 cm scoring cylinder. "
"Source photons (both lines, 50:50) sampled uniformly in solid angle of the 10x10 cm pyramid; uncollided primaries scored analytically with exponential attenuation. "
"Every phantom collision (Klein-Nishina Compton, Rayleigh, pair with two 511 keV isotropic annihilation photons, all transported) gives candidate interactions in the cylinder weighted by cross-section fractions and 1/r^2 geometry. "
"200 events drawn by systematic probability-proportional-to-weight sampling without replacement; process at the point chosen by cross-section fraction, kinematics sampled from KN/Rayleigh/pair models.")
def r(x,n=3):
    return None if x is None else round(float(x),n)
out=[]
for e in ev:
    out.append({"process":e['process'],"e_in_MeV":r(e['e_in_MeV']),"e_out_MeV":r(e['e_out_MeV']),
                "cos_scatter":r(e['cos_scatter']),"t_e_MeV":r(e['t_e_MeV']),
                "cos_electron":r(e['cos_electron']),"t_positron_MeV":r(e['t_positron_MeV'])})
s=json.dumps({"method":method,"events":out},separators=(',',':'))
open('final.json','w').write(s)
print(len(s), len(out))
lines=[json.dumps(o,separators=(',',':')) for o in out]
open('lines.txt','w').write('\n'.join(lines))
'''
open('fin2.py','w').write(t)
r=subprocess.run(['python','fin2.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-3000:])
print(open('lines.txt').read()[:20000])
