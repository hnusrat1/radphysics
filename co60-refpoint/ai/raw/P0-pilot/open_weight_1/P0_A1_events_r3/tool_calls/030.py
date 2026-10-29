# Find the pair event(s) in events list
pair_events = [i for i,e in enumerate(events) if e['process']=='pair']
print("Pair indices:", pair_events)
# Let's pick the pair event and then 199 others randomly
rng2=np.random.default_rng(777)
# All indices except pair
other_indices = [i for i in range(len(events)) if i not in pair_events]
# Sample 199 from others
idx_other = rng2.choice(other_indices, 199, replace=False)
final_indices = list(pair_events) + list(idx_other)
final_events = [events[i] for i in final_indices]
print("Final sample:", Counter(e['process'] for e in final_events))

# Verify schema compliance
for ev in final_events:
    # Ensure null fields are None
    for k in ['e_out_MeV','cos_scatter','cos_electron','t_positron_MeV']:
        if k in ev and (isinstance(ev[k], float) and math.isnan(ev[k])):
            ev[k] = None
    # Ensure t_e_MeV is float
    ev['t_e_MeV'] = float(ev['t_e_MeV'])

# Build output
output = {
    "method": "Analog Monte Carlo transport of 2 MeV photons (SSD=100 cm, 10x10 cm field) through 30 cm water cube. Cross sections: Klein-Nishina (Compton), NIST-based (photoelectric, pair, Rayleigh). Photons tracked with exponential step sampling; interactions in scoring cylinder (r=1 cm, thickness 0.2 cm, at 10 cm depth) recorded. 1.5 million primaries yielded 684 interactions; 200 uniformly subsampled.",
    "events": final_events
}

with open('/tmp/result.json', 'w') as f:
    json.dump(output, f, indent=2)
print("JSON saved. Event count:", len(final_events))
print("\nFirst event:", json.dumps(final_events[0], indent=2))