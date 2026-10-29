
import subprocess
r=subprocess.run(['python','/tmp/work/mc.py','12','10000000'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-150:],r.stderr[-500:])
print(os.path.exists('/tmp/work/out_12.pkl'))
