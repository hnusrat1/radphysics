import subprocess
p='/tmp/mc.cpp';r=subprocess.run(['g++','-O3','-std=c++17',p,'-o','/tmp/mc'],capture_output=True,text=True);print(r.stderr,r.returncode)
