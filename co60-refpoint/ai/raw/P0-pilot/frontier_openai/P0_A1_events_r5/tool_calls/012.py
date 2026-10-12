import numpy as np, random, json, collections
arr=np.loadtxt('/tmp/e.csv'); random.seed(930214); ix=random.sample(range(len(arr)),200); samp=arr[ix]
names={0:'compton',1:'photoelectric',2:'pair',3:'rayleigh'}
events=[]
for r in samp:
 p=int(r[0]); Ein,Eout,cs,Te,ce,Tp=r[1:]
 rd=lambda x:round(float(x),8)
 events.append({'process':names[p],'e_in_MeV':rd(Ein),'e_out_MeV':None if p in (1,2) else rd(Eout),'cos_scatter':None if p in (1,2) else rd(cs),'t_e_MeV':rd(Te),'cos_electron':None if p==3 else rd(ce),'t_positron_MeV':None if p!=2 else rd(Tp)})
open('/tmp/events.json','w').write(json.dumps(events,separators=(',',':')))
print(collections.Counter(e['process'] for e in events),len(events))