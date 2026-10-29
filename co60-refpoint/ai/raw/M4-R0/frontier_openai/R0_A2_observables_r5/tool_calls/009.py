s=open('/tmp/mc.cpp').read();s=s.replace('double totTrack[2]={}; long long', 'double totTrack[2]={}, eTrack[2]={}; long long')
s=s.replace('totTrack[cl]+=l;int eb=', 'totTrack[cl]+=l;eTrack[cl]+=l*energy;int eb=')
s=s.replace('cout<<"factor "<<factor<<" primary "<<totTrack[0]*factor<<" scatter "<<totTrack[1]*factor<<"\\n";', 'cout<<"factor "<<factor<<" primary "<<totTrack[0]*factor<<" scatter "<<totTrack[1]*factor<<" Etrack "<<eTrack[0]*factor<<" "<<eTrack[1]*factor<<"\\n";')
s=s.replace('cout<<"EDIR ";for(int i=0;i<10;i++)cout<<edir[i][0]<<",";cout<<"\\n";', 'cout<<"EDIR ";for(int i=0;i<10;i++)cout<<edir[i][0]+edir[i][1]<<",";cout<<"\\nEDIRP ";for(int i=0;i<10;i++)cout<<edir[i][0]<<",";cout<<"\\n";')
open('/tmp/mc.cpp','w').write(s)
import subprocess
r=subprocess.run(['g++','-O3','-std=c++17','/tmp/mc.cpp','-o','/tmp/mc'],capture_output=True,text=True);print(r.stderr)
