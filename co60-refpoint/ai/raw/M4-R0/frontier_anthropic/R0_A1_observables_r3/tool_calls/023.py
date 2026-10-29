
s=open('/tmp/work/full.py').read()
s=s.replace("""        kern=np.exp(-x/m); kern=kern/(np.sum(kern)*dz)
        Dz[i]=np.sum(Kp[mask]*kern*dz)""","""        kern=np.exp(-x/m)/m
        wts=np.ones(kern.size)*dz; wts[0]*=0.5; wts[-1]*=0.5
        Dz[i]=np.sum(Kp[mask]*kern*wts)""")
s=s.replace("for _m in [0.07,0.1,0.13]:","for _m in [0.07,0.1,0.13,0.2]:")
open('/tmp/work/full.py','w').write(s)
import subprocess, pickle, numpy as np
r=subprocess.run(['python3','/tmp/work/full.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-800:], r.stderr[-2000:])
out=pickle.load(open('/tmp/work/out.pkl','rb'))
for k in ['L5_D_over_phi0_Gy_cm2','L5_beta','L5_PDD5_percent','_dmax','L5_Kcol_over_phi0_Gy_cm2','L5_K_over_phi0_Gy_cm2','L5_g']:
    print(k,out[k])
