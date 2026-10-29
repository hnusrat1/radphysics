
import subprocess
t=r'''
import json
d=json.load(open('events.json'))
ev=d['events']
method=("Analog photon Monte Carlo in the water cube with next-event estimation to the scoring cylinder. "
"Source photons (1.1732/1.3325 MeV, 50:50) sampled uniformly in solid angle of the 10x10 cm pyramid; "
"primary interactions in the cylinder scored analytically (uncollided fluence with exp attenuation). "
"Every collision in the phantom (Compton via Klein-Nishina, Rayleigh with form-factor angles, pair with 2x511 keV isotropic annihilation photons, transported in turn) "
"gives candidate interactions in the cylinder with weights from Compton/Rayleigh/pair cross sections and 1/r^2 geometry. "
"200 events drawn by systematic probability-proportional-to-weight sampling without replacement; process at the point chosen by cross-section fractions, kinematics sampled from KN/Rayleigh/pair models.")
print(len(method.split()))
out=[]
def r(x,n=5):
    return None if x is None else round(float(x),n)
for e in ev:
    out.append({"process":e['process'],"e_in_MeV":r(e['e_in_MeV'],4),"e_out_MeV":r(e['e_out_MeV'],4),
                "cos_scatter":r(e['cos_scatter'],4),"t_e_MeV":r(e['t_e_MeV'],4),
                "cos_electron":r(e['cos_electron'],4),"t_positron_MeV":r(e['t_positron_MeV'],4)})
obj={"method":method,"events":out}
s=json.dumps(obj,separators=(',',':'))
open('final.json','w').write(s)
print(len(s))
print(s)
'''
open('fin.py','w').write(t)
r=subprocess.run(['python','fin.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-3000:])
