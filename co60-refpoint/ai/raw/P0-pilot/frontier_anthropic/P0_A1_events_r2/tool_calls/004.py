
import subprocess
r = subprocess.run(['python3','sim.py','3000000'],capture_output=True,text=True,cwd='.')
print(r.stdout, r.stderr[-2000:])
