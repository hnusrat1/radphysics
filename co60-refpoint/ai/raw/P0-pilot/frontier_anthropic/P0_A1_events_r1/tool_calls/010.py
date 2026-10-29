
import subprocess
code=open('mc.py').read()
dbg = code.replace('    vol=in_volume(pos)\n','    vol=in_volume(pos)\n    print("it",it,pos.shape[0],"vol",vol.sum(), "zmean",pos[:,2].mean(), "E",en.mean(), file=sys.stderr)\n')
open('mc_dbg.py','w').write(dbg)
r=subprocess.run(['python','mc_dbg.py','1','20000'],capture_output=True,text=True)
print(r.stderr[:3000], r.stdout)
