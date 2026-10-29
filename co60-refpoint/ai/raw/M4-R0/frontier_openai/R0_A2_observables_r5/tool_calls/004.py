p='/tmp/mc.cpp';s=open(p).read();s=s.replace('V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}', 'V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};} V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}')
r=subprocess.run(['g++','-O3','-std=c++17',p,'-o','/tmp/mc'],capture_output=True,text=True);print(r.stderr,r.returncode)
