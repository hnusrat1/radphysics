import json
d=json.load(open("answer.json"))
print(json.dumps(d, separators=(",",":")))