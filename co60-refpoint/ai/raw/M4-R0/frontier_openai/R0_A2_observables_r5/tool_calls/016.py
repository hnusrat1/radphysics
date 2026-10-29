s=open('/tmp/mc.cpp').read();old='cout<<"EDIR ";for(int i=0;i<10;i++)cout<<edir[i][0]<<",";cout<<"\\n";'
print(old in s)
s=s.replace(old,'cout<<"EDIR ";for(int i=0;i<10;i++)cout<<edir[i][0]+edir[i][1]<<",";cout<<"\\nEDIRP ";for(int i=0;i<10;i++)cout<<edir[i][0]<<",";cout<<"\\nEDIRS ";for(int i=0;i<10;i++)cout<<edir[i][1]<<",";cout<<"\\n";')
open('/tmp/mc.cpp','w').write(s)
import subprocess
r=subprocess.run(['g++','-O3','-std=c++17','/tmp/mc.cpp','-o','/tmp/mc'],capture_output=True,text=True);print(r.stderr)
