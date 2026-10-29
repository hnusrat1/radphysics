
import subprocess, json, collections
tot=[]
for sd in range(2,22):
    r=subprocess.run(['python','sim.py',str(sd),'2000000'],capture_output=True,text=True,cwd='.')
    tot+=json.load(open(f'ev_{sd}.json'))
print(len(tot))
print(collections.Counter(e[0] for e in tot))
