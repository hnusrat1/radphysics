events = []
t0 = time.time()
run_batch(50000, events)
t1 = time.time()
print(f"50000 primaries: {len(events)} events in {t1-t0:.1f}s")
if events:
    cnt = Counter(e['process'] for e in events)
    print(cnt)
    print(events[0])