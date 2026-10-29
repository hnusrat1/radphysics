# Quick test: count secondary events in the sample
events, prim = simulate_directed(200, 50000)
primary_events = [e for e in events if e['e_in_MeV'] == 2.0]
scattered_events = [e for e in events if e['e_in_MeV'] != 2.0]
print("Primary events:", len(primary_events))
print("Scattered events:", len(scattered_events))
if scattered_events:
    print("Example:", scattered_events[0])