
import json,collections
ev=json.load(open('events_1.json'))
print(len(ev)); print(collections.Counter(e['process'] for e in ev))
