p='/tmp/mc.cpp';s=open(p).read();s=s.replace('double proc[2][4]={},trans[2][4]={},elcnt[2]={},elsum[2]={},ebin[2][25]={},edirect[10]={};','double proc[2][4]={},trans[2][4]={},elcnt[2]={},elsum[2]={},allcnt[2]={},allsum[2]={},ebin[2][25]={},edirect[10]={};')
s=s.replace('de=E*(1-ratio);double phi=', 'de=E*(1-ratio);allcnt[par]++;allsum[par]+=de;double phi=')
s=s.replace('}else if(typ==1){de=E;int eb=', '}else if(typ==1){de=E;allcnt[par]++;allsum[par]+=de;int eb=')
s=s.replace('de=max(0.0,E-1.021998);if(de>0){for(int j=0;j<2;j++){', 'de=max(0.0,E-1.021998);if(de>0){allcnt[par]+=2;allsum[par]+=de;for(int j=0;j<2;j++){')
s=s.replace('<<" EL "<<elcnt[j]<<" "<<elsum[j]<<" EB "','<<" EL "<<elcnt[j]<<" "<<elsum[j]<<" ALL "<<allcnt[j]<<" "<<allsum[j]<<" EB "')
open(p,'w').write(s)
import subprocess
p=subprocess.run(['g++','-O3','-std=c++17','/tmp/mc.cpp','-o','/tmp/mc'],capture_output=True,text=True);print(p.stderr)
