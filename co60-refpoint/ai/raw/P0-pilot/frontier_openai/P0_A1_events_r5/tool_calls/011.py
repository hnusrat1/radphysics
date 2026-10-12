import subprocess
subprocess.run(['g++','-O3','-std=c++17','/tmp/mc.cpp','-o','/tmp/mc'],check=True)
r=subprocess.run(['/tmp/mc'],capture_output=True,text=True,timeout=600)
print(r.stderr)
open('/tmp/e.csv','w').write(r.stdout)