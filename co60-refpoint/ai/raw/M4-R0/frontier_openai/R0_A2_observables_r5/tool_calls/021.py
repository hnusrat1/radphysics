import subprocess,time
r=subprocess.run(['/tmp/mc','30000000'],capture_output=True,text=True,timeout=590);print(r.stderr);print(r.stdout[-500:])