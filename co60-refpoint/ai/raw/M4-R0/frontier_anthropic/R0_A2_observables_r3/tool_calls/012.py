
import subprocess
r=subprocess.run("cd /tmp/work && python driver.py 16 1250000 t1",shell=True,capture_output=True,text=True)
print(r.stdout, r.stderr[-2000:])
