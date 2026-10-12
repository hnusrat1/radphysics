
import subprocess
r=subprocess.run(['python','t2.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
