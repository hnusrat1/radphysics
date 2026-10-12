import json
E=json.load(open('/tmp/events.json'))
for chunk in range(4):
 print('CHUNK',chunk+1)
 print(',\n'.join(json.dumps(x,separators=(',',':')) for x in E[chunk*50:(chunk+1)*50]))