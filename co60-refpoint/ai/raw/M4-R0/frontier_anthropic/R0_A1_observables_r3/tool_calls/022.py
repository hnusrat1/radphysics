
s=open('/tmp/work/full.py').read()
s=s.replace("""        Dz[i]=np.sum(Kp[mask]*np.exp(-x/m)/m*dz)""","""        kern=np.exp(-x/m); kern=kern/(np.sum(kern)*dz)
        Dz[i]=np.sum(Kp[mask]*kern*dz)""")
s=s.replace("MKERN=0.10","MKERN=0.10\nfor _m in [0.07,0.1,0.13]:\n    _Dz=Dconv(_m,zg); _i=np.argmax(_Dz); print('scan m',_m,'dmax',round(zg[_i],3),'PDD5',round(100*Dconv(_m,np.array([5.0]))[0]/_Dz[_i],2),'beta5',round(Dconv(_m,np.array([5.0]))[0]/Kint(5.0)[0],4))")
open('/tmp/work/full.py','w').write(s)
import subprocess
r=subprocess.run(['python3','/tmp/work/full.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-800:], r.stderr[-2000:])
out=pickle.load(open('/tmp/work/out.pkl','rb'))
for k in ['L5_D_over_phi0_Gy_cm2','L5_beta','L5_PDD5_percent','_dmax','L5_Kcol_over_phi0_Gy_cm2','L5_K_over_phi0_Gy_cm2','L5_g']:
    print(k,out[k])
