
import pickle, json, numpy as np
L=pickle.load(open('/tmp/work/final.pkl','rb'))
def r(x,n=5):
    if isinstance(x,list): return [r(y,n) for y in x]
    if isinstance(x,dict): return {k:r(v,n) for k,v in x.items()}
    return float(f"{x:.{n}g}")
for k in ['L1_photon_fluence_spectrum_primary','L1_photon_fluence_spectrum_scattered','L1_angular_primary','L1_angular_scattered','L2_interaction_fractions','L2_energy_transfer_fractions','L2_interaction_fractions_primary','L2_interaction_fractions_scattered','L3_initial_spectrum','L3_initial_spectrum_parent_primary','L3_initial_direction','L4_electron_fluence_spectrum','L4_angular']:
    print(k, json.dumps(r(L[k],4)))
