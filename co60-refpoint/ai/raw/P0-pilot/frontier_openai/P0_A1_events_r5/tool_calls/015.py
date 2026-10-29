import json
x=json.load(open('/tmp/events.json'))
for i,e in enumerate(x[50:75],51): print(json.dumps(e,separators=(',',':'))+(',' if i<75 else ''))