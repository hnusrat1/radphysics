import random, json, math
random.seed(930214)
idx=random.sample(range(len(arr)),200)
samp=arr[idx]
print(collections.Counter(samp[:,0].astype(int)))
# convert records according schema
names={0:'compton',1:'photoelectric',2:'pair',3:'rayleigh'}
events=[]
for row in samp:
 p=int(row[0]); Ein,Eout,cs,Te,ce,Tp=row[1:]
 def rnd(x): return round(float(x),8)
 ev={'process':names[p], 'e_in_MeV':rnd(Ein),
     'e_out_MeV':None if p==1 or p==2 else rnd(Eout),
     'cos_scatter':None if p==1 or p==2 else rnd(cs),
     't_e_MeV':rnd(Te),
     'cos_electron':None if p==3 else rnd(ce),
     't_positron_MeV':None if p!=2 else rnd(Tp)}
 events.append(ev)
print(json.dumps(events,indent=2))