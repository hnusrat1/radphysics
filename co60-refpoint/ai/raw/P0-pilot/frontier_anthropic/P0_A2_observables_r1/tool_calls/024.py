
s=open('/tmp/work/final2.py').read()
s=s.replace("tb=ER['tb']; tcos=ER['tcos']; tt_=ER['tot']","tb=ER['tb'][:25]; tcos=ER['tcos']; tt_=ER['tot']")
open('/tmp/work/final2.py','w').write(s)
import subprocess
r=subprocess.run(['python3','/tmp/work/final2.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-6000:], r.stderr[-3000:])
