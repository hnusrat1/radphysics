import subprocess
p='/tmp/mc.cpp'
s=open(p).read().replace('if(x<0||x>30||y < -15||y>15||z<0||z>30)', 'if(x < -15||x > 15||y < -15||y>15||z<0||z>30)').replace('while(ur()>pow(1+u*u+v*v,1.5))','while(ur()>pow(1+u*u+v*v,-1.5))')
open(p,'w').write(s)
subprocess.run(['g++','-O3','-std=c++17',p,'-o','/tmp/mc'],check=True)
r=subprocess.run(['/tmp/mc'],capture_output=True,text=True,timeout=600)
print(r.stderr, len(r.stdout), r.returncode)
open('/tmp/e.csv','w').write(r.stdout)
