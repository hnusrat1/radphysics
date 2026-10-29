
import json
d=json.load(open('answer.json'))
s=json.dumps(d['events'],separators=(',',':'))
print(len(s))
print(s[:12000])
