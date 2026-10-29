
import subprocess
r=subprocess.run(['python','sample.py'],capture_output=True,text=True); print(r.stdout, r.stderr[-2000:])
