import json
x=json.load(open('/tmp/events.json'))
for i,e in enumerate(x[25:50],26): print(json.dumps(e,separators=(',',':'))+(',' if i<50 else ''))