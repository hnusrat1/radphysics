
import subprocess
r=subprocess.run(['python','sim.py','20000'],capture_output=True,text=True,cwd='.')
print(r.stdout, r.stderr[-3000:])
