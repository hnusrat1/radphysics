import json
with open("answer.json","r") as f:
    output = json.load(f)
print(json.dumps(output, indent=2)[:5000])
print("...")