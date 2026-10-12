events = []
t0 = time.time()
run_batch(1000000, events)
t1 = time.time()
print(f"1e6 primaries: {len(events)} events in {t1-t0:.1f}s")
if events:
    cnt = Counter(e['process'] for e in events)
    print(cnt)
    # show some stats
    e_in = [e['e_in_MeV'] for e in events]
    print("E_in range:", min(e_in), max(e_in))
    print("E_in mean:", np.mean(e_in))
    procs = [e['process'] for e in events]
    for p in ['compton','photoelectric','pair','rayleigh']:
        sub = [e for e in events if e['process']==p]
        print(f"{p}: {len(sub)} events")