import numpy as np
# Using events from previous run
e_in = np.array([e['e_in_MeV'] for e in events])
print("Energy quantiles:")
for q in [0,10,25,50,75,90,100]:
    print(f"  {q}%: {np.percentile(e_in,q):.4f} MeV")
print("Mean:", np.mean(e_in))
print("Median:", np.median(e_in))

# For each process, show energy range
for proc in ['compton','photoelectric','pair','rayleigh']:
    sub = [e for e in events if e['process']==proc]
    if sub:
        es = [e['e_in_MeV'] for e in sub]
        print(f"{proc}: n={len(sub)}, e_in range {min(es):.4f}-{max(es):.4f}")

# Check that compton angles are reasonable
comps = [e for e in events if e['process']=='compton']
cos_sc = [e['cos_scatter'] for e in comps]
cos_el = [e['cos_electron'] for e in comps]
print("Compton cos_scatter range:", min(cos_sc), max(cos_sc))
print("Compton cos_electron range:", min(cos_el), max(cos_el))
print("Compton e_out range:", min(e['e_out_MeV'] for e in comps), max(e['e_out_MeV'] for e in comps))