
chk=r'''
import sys; sys.path.insert(0,'/tmp/work')
import numpy as np, pickle
def Sbethe(Tk, ZA=0.55508, I=75e-6):
    mc2=0.51099895
    t=Tk/mc2; b2=t*(t+2)/(t+1)**2
    F=(1-b2)*(1+t*t/8-(2*t+1)*np.log(2))
    return 0.153537*ZA/b2*(np.log(t*t*(t+2)/(2*(I/mc2)**2))+F)
for T in [0.1,0.5,1.0,2.0]:
    print(T, round(float(Sbethe(T)),4))
R=pickle.load(open('/tmp/work/R.pkl','rb'))
print(R['_dmax_cm'], R['_PDD'][:20])
'''
open('/tmp/work/chk.py','w').write(chk)
import subprocess
r=subprocess.run(['python','/tmp/work/chk.py'],capture_output=True,text=True)
print(r.stdout, r.stderr[-800:])
