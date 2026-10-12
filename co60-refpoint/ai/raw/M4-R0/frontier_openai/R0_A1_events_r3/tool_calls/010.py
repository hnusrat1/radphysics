events=[]
for l in open('/tmp/sample.txt'):
 a=l.split(); p=int(a[0]); Ein,Eout,cs,Te,ce,Tp=map(float,a[1:])
 ev={'process':procmap[p], 'e_in_MeV':Ein, 'e_out_MeV':None if p in (1,2) else Eout, 'cos_scatter':None if p in (1,2) else cs, 't_e_MeV':Te, 'cos_electron':None if p==3 else ce, 't_positron_MeV':None if p!=2 else Tp}
 events.append(ev)
print(len(events),events[0],events[1]);
from collections import Counter
print(Counter(e['process'] for e in events));
open('/tmp/events.json','w').write(json.dumps(events,indent=2))