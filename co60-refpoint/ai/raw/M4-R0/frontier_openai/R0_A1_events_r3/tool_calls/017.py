import json
E=json.load(open('/tmp/events.json'))
print(',\n'.join(json.dumps(x,separators=(',',':')) for x in E[50:100]))