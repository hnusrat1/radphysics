import subprocess,time
st=time.time();r=subprocess.run(['/tmp/mc','50000000'],capture_output=True,text=True,timeout=590);print(time.time()-st); print(r.stderr); print(r.stdout)