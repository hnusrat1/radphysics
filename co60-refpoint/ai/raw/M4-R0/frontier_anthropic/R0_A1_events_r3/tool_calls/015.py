import json
d=json.load(open("answer.json"))
for e in d["events"][50:110]:
    print(json.dumps(e,separators=(",",":")))
