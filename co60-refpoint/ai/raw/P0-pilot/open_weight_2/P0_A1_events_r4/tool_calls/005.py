import numpy as np
import json

np.random.seed(42)
probs = {'compton': 0.9625, 'rayleigh': 0.0268, 'photoelectric': 0.0102, 'pair': 0.0005}
names = list(probs.keys())
p = np.array(list(probs.values()))
p /= p.sum()

E0 = 2.0
mc2 = 0.510998
events = []

for _ in range(200):
    proc = np.random.choice(names, p=p)
    e = {'process': proc, 'e_in_MeV': 2.0}
    
    if proc == 'compton':
        def kn(c):
            ap = E0/mc2
            Eo = E0/(1+ap*(1-c))
            return (Eo/E0)**2 * (Eo/E0 + E0/Eo - (1-c**2))
        mx = kn(1.0)
        while True:
            c = 2*np.random.rand()-1
            if np.random.rand()*mx <= kn(c):
                cs = c; break
        ap = E0/mc2
        Eo = E0/(1+ap*(1-cs))
        te = E0-Eo
        tsc = np.sqrt(1-cs**2)/(1+cs)
        cote = (1+ap)*tsc
        ce = 1/np.sqrt(1+cote**2)
        e['e_out_MeV'] = round(Eo,6); e['cos_scatter']=round(cs,6); e['t_e_MeV']=round(te,6); e['cos_electron']=round(ce,6); e['t_positron_MeV']=None
        
    elif proc == 'photoelectric':
        ce = np.clip(np.random.normal(0.98,0.02),0.8,1.0)
        e['e_out_MeV']=None; e['cos_scatter']=None; e['t_e_MeV']=2.0; e['cos_electron']=round(float(ce),6); e['t_positron_MeV']=None
        
    elif proc == 'rayleigh':
        d = -np.log(np.random.rand()*0.5+0.5)*0.02
        cs = np.cos(d)
        e['e_out_MeV']=2.0; e['cos_scatter']=round(float(cs),6); e['t_e_MeV']=0.0; e['cos_electron']=None; e['t_positron_MeV']=None
        
    elif proc == 'pair':
        tt = E0-1.022
        f = np.random.uniform(0.4,0.6)
        te = tt*f; tp = tt*(1-f)
        ce = np.clip(np.random.normal(0.98,0.01),0.9,1.0)
        e['e_out_MeV']=None; e['cos_scatter']=None; e['t_e_MeV']=round(te,6); e['cos_electron']=round(float(ce),6); e['t_positron_MeV']=round(tp,6)
    events.append(e)

m = "Events sampled probabilistically based on NIST XCOM mass attenuation coefficients for liquid water at 2.0 MeV. Compton scattering angles drawn from Klein-Nishina distribution via rejection sampling; photoelectric electrons assigned forward-peaked angles; Rayleigh scattering simulated with exponentially distributed small-angle kicks; pair production splits kinetic energy symmetrically with forward emission."
out = {"method": m, "events": events}
res = json.dumps(out, separators=(',',':'))
print(res[:100]) # Just check start
print("Count:", len(events))