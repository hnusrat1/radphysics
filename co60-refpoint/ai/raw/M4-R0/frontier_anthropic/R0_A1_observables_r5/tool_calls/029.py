
import subprocess, sys
def runpy(path):
    r=subprocess.run([sys.executable,path],capture_output=True,text=True,cwd='/tmp/work')
    return r.stdout, r.stderr[-3000:]
o,e=runpy('/tmp/work/an2.py'); print(o[-5000:], e[-3000:])
