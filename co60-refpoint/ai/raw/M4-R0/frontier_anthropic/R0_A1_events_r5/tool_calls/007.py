
final = r'''
import time, json
exec(open("mc.py").read())
t=time.time()
ev = run(4000000)
print("time", time.time()-t, "N events in volume", len(ev))
from collections import Counter
print(Counter(e['process'] for e in ev))
import pickle
pickle.dump(ev, open("events_all.pkl","wb"))
'''
open('final.py','w').write(final)
r=subprocess.run(['python','final.py'],capture_output=True,text=True); print(r.stdout, r.stderr[-3000:])
