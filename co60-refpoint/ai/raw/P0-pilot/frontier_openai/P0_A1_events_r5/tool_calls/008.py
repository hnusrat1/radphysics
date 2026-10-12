import numpy as np,collections,json
arr=np.loadtxt('/tmp/e.csv'); ev=json.load(open('/tmp/events.json'))
print(len(ev),collections.Counter(e['process'] for e in ev))
print('count primary',sum(e['e_in_MeV']==2 for e in ev))
for i in range(0,len(ev),25):
 print(f'CHUNK {i//25+1}')
 print('\n'.join(json.dumps(x,separators=(',',':'))+',' for x in ev[i:i+25]))