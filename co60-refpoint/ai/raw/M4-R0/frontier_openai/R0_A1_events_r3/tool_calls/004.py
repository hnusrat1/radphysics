import subprocess
r=subprocess.run(['g++','-O3','-std=c++17','/tmp/sim.cpp','-o','/tmp/sim'],capture_output=True,text=True); print(r.returncode,r.stderr[:2000])