import json,collections
x=json.load(open('/tmp/events.json'))
print(len(x),collections.Counter(e['process'] for e in x),sum(e['e_in_MeV']==2.0 for e in x))