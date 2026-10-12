import subprocess,time
p=subprocess.run(['/tmp/mc','30000000'],capture_output=True,text=True,timeout=550);print(p.stdout)