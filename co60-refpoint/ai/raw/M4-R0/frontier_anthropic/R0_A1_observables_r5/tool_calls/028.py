
import subprocess, sys
def runpy(path):
    r=subprocess.run([sys.executable,path],capture_output=True,text=True,cwd='/tmp/work')
    return r.stdout, r.stderr[-3000:]
code = open('/tmp/work/an2.py').read()
print('Stot=Scol+Srad' in code, 'Mdel' in code)
