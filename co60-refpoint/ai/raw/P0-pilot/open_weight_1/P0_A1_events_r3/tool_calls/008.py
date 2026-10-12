events = []
t0 = time.time()
simulate_batch(200000, rng, events)
t1 = time.time()
print(f"200000 primaries: {len(events)} events in {t1-t0:.1f}s")
if events:
    from collections import Counter
    cnt = Counter(e['process'] for e in events)
    print("Process counts:", cnt)
    # Print a few samples
    for e in events[:5]:
        print(e)