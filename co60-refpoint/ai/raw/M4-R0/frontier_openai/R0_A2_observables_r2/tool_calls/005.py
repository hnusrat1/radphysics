p='/tmp/mc.cpp'; s=open(p).read(); s=s.replace('double mu;do{mu=2*R()-1;double ratio=1/(1+(E/.51099895)*(1-mu));}while(R()>0.5*pow(1/(1+(E/.51099895)*(1-mu)),2)*(1/(1+(E/.51099895)*(1-mu))+1+(1+(E/.51099895)*(1-mu))-(1-mu*mu)));','double mu,rr;do{mu=2*R()-1;rr=1/(1+(E/.51099895)*(1-mu));}while(R()>0.5*rr*rr*(rr+1/rr-(1-mu*mu)));')
s=s.replace('if(typ!=3)isprim=false;','isprim=false;')
open(p,'w').write(s)
import subprocess
p=subprocess.run(['g++','-O3','-std=c++17','/tmp/mc.cpp','-o','/tmp/mc'],capture_output=True,text=True);print(p.stderr)
