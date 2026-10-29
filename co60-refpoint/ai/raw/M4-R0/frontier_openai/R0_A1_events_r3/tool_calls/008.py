import json
procmap={0:'compton',1:'photoelectric',2:'pair',3:'rayleigh'}
events=[]
for l in open('/tmp/sample.txt'):
 a=l.split(); p=int(a[0]); vals=list(map(float,a[1:]))
 ev={'process':procmap[p], 'e_in_MeV':vals[0], 'e_out_MeV':None if vals[1]<0 else vals[1], 'cos_scatter':None if vals[2]<0 else vals[2], 't_e_MeV':vals[3], 'cos_electron':None if vals[4]<0 else vals[4], 't_positron_MeV':None if vals[5]<0 else vals[5]}
 events.append(ev)
print(len(events)); print(json.dumps(events, separators=(',',':'))[:1500]); open('/tmp/events.json','w').write(json.dumps(events,indent=2))