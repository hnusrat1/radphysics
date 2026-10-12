import json
x=json.load(open('/tmp/events.json'))
for i,e in enumerate(x[175:200],176): print(json.dumps(e,separators=(',',':'))+(',' if i<200 else ''))