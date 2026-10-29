import json
x=json.load(open('/tmp/events.json'))
for i,e in enumerate(x[75:100],76): print(json.dumps(e,separators=(',',':'))+(',' if i<100 else ''))