s=open('/tmp/mc.cpp').read();s=s.replace('V operator*(V a,double s)', 'V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};} V operator*(V a,double s)');open('/tmp/mc.cpp','w').write(s)
import subprocess
r=subprocess.run(['g++','-O3','-std=c++17','/tmp/mc.cpp','-o','/tmp/mc'],capture_output=True,text=True);print(r.stderr,r.returncode)