
import numpy as np
import json

me = 0.510998950
mu_compton = 0.0428
mu_photoelectric = 0.0001
mu_rayleigh = 0.0002
mu_pair = 0.0052
mu_total = mu_compton + mu_photoelectric + mu_rayleigh + mu_pair
probs = [mu_compton/mu_total, mu_photoelectric/mu_total, mu_pair/mu_total, mu_rayleigh/mu_total]
n_events = 200

np.random.seed(42)
counts = np.random.multinomial(n_events, probs)

def sample_kn(E):
    alpha = E / me
    for _ in range(1000):
        ct = 2*np.random.rand()-1
        Ep = E/(1+alpha*(1-ct))
        s2 = 1-ct**2
        f = (Ep/E)**2 * (E/Ep + Ep/E - s2)
        cts = np.linspace(-1,1,200)
        fs = [(E/(1+alpha*(1-c)))**2 * ((1+alpha*(1-c)) + (1/(1+alpha*(1-c))) - (1-c**2)) for c in cts]
        if np.random.rand()*max(fs) <= f:
            return ct, Ep
    return 0.0, E/(1+alpha)

events = []
# compton
for _ in range(counts[0]):
    ct, Ep = sample_kn(2.0)
    events.append({"process":"compton","e_in_MeV":2.0,"e_out_MeV":round(Ep,6),"cos_scatter":round(ct,6),
                   "t_e_MeV":round(2.0-Ep,6),"cos_electron":None,"t_positron_MeV":None})

# photoelectric
for _ in range(counts[1]):
    ue = 2.0-0.00053
    u = np.random.rand()
    ce = 1.0 - 2.0*u/(1.0+u)
    events.append({"process":"photoelectric","e_in_MeV":2.0,"e_out_MeV":None,"cos_scatter":None,
                   "t_e_MeV":round(ue,6),"cos_electron":round(np.clip(ce,-1,1),6),"t_positron_MeV":None})

# pair
for _ in range(counts[2]):
    exc = 2.0 - 2*me
    fp = np.random.uniform(0.35,0.65)
    Tpos = fp*exc
    Tel = exc-Tpos
    cep = np.random.beta(1.5,0.5)*2-1
    ce = np.random.beta(1.5,0.5)*2-1
    events.append({"process":"pair","e_in_MeV":2.0,"e_out_MeV":None,"cos_scatter":None,
                   "t_e_MeV":round(Tel,6),"cos_electron":round(ce,6),"t_positron_MeV":round(Tpos,6)})

# rayleigh
for _ in range(counts[3]):
    cr = 1.0 - np.random.exponential(0.05)
    cr = max(0.0,min(1.0,cr))
    events.append({"process":"rayleigh","e_in_MeV":2.0,"e_out_MeV":2.0,"cos_scatter":round(cr,6),
                   "t_e_MeV":0.0,"cos_electron":None,"t_positron_MeV":None})

np.random.shuffle(events)
assert len(events)==200
assert all(e["t_e_MeV"] is not None for e in events)

out = {"method":"Random sampling at 2.0 MeV using NIST XCOM cross sections. Compton angles from Klein-Nishina; pair energy sharing uniform; photoelectric angular distribution forward-peaked; Rayleigh coherent scatter forward-peaked.", "events":events}
print(json.dumps(out))
