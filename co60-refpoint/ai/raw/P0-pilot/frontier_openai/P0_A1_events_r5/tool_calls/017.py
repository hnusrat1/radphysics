import json
x=json.load(open('/tmp/events.json'))
for i,e in enumerate(x[100:125],101): print(json.dumps(e,separators=(',',':'))+(',' if i<125 else ''))