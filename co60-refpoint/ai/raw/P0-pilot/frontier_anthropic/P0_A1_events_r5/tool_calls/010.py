
import json
d=json.load(open('/tmp/work/events.json'))
ev=d['events']
def fmt(e):
    return json.dumps(e,separators=(',',':'))
lines=[fmt(e) for e in ev]
print("\n".join(lines[:100]))
