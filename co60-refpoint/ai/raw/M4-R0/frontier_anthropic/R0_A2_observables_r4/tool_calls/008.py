
import sys; sys.path.insert(0,'/tmp/work')
import subprocess
print(subprocess.run(['python','ana.py'],capture_output=True,text=True,cwd='/tmp/work'))
