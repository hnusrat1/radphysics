import json
x=json.load(open('/tmp/events.json'))
for i,e in enumerate(x[150:175],151): print(json.dumps(e,separators=(',',':'))+(',' if i<175 else ''))