import json
x=json.load(open('/tmp/events.json'))
for i,e in enumerate(x[125:150],126): print(json.dumps(e,separators=(',',':'))+(',' if i<150 else ''))