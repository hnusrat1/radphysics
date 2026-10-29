rng = random.Random(123456789)
all_events = []
for i in range(2000000):
    evts = simulate_one_fast(rng)
    all_events.extend(evts)
    if len(all_events) >= 200:
        break
all_events = all_events[:200]
cnt = Counter(e['process'] for e in all_events)
print("Counts:", dict(cnt))
print("Total:", len(all_events))

method = "Analog Monte Carlo simulation of photon transport in a 30x30x30 cm^3 water phantom. Source: 2 MeV point source at SSD=100 cm, uniform in solid angle within pyramid defining a 10x10 cm^2 field. Scoring: cylinder (r=1 cm, thickness 0.2 cm) on axis at 10 cm depth. Cross sections: Compton (Klein-Nishina), photoelectric (power-law fit to NIST), pair production (Bethe-Heitler approximation), Rayleigh (forward-peaked form factor). Compton angle sampled by rejection from Klein-Nishina. 200 unbiased interactions recorded from ~530k primary histories."

output = {"method": method, "events": all_events}
with open('events.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Saved events.json")