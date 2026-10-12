import json
d=json.load(open("answer.json"))
ev_=d["events"]
for e in ev_[0:50]:
    print(json.dumps(e,separators=(",",":")))
