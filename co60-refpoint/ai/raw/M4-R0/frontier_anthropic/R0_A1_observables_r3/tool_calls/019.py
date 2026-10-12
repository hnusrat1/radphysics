
s=open('/tmp/work/full.py').read()
s=s.replace("def Kint(z):\n    return np.interp(z,zt,Kcol_z)   # zt starts at 0.05 (constant below)",
"""_msk=zt>=0.5
_coef=np.polyfit(zt[_msk],np.log(Kcol_z[_msk]),3)
def Kint(z):
    z=np.atleast_1d(z).astype(float)
    smooth=np.exp(np.polyval(_coef,z))
    raw=np.interp(z,zt,Kcol_z)
    return np.where(z>=0.5,smooth,raw)""")
s=s.replace("""for m in [0.1,0.15,0.2,0.3]:
    Dz=Dconv(m,zg)
    imax=np.argmax(Dz)
    D5=Dconv(m,np.array([5.0]))[0]
    pdd5=100*D5/Dz[imax]
    print("m",m,"dmax",round(zg[imax],3),"D(dmax)/K(dmax)",round(Dz[imax]/Kint(zg[imax]),4),"beta5",round(D5/Kint(5.0),4),"PDD5",round(pdd5,2), "D(0.05)/Dmax", round(Dz[0]/Dz[imax],3))
out['_scan']=None""","""MKERN=0.10
Dz=Dconv(MKERN,zg); imax=np.argmax(Dz); dmax=zg[imax]
D5=Dconv(MKERN,np.array([5.0]))[0]
beta5=D5/Kint(5.0)[0]
out['L5_D_over_phi0_Gy_cm2']=float(D5*conv)
out['L5_beta']=float(beta5)
out['L5_PDD5_percent']=float(100*D5/Dz[imax])
out['_dmax']=float(dmax)
out['_Kint5']=float(Kint(5.0)[0]*conv)
out['_Kcol_z']=dict(zip([float(z) for z in zpts],[float(x) for x in Kcol_z]))
# Kerma-based K_col at 5 cm
out['L5_Kcol_over_phi0_Gy_cm2']=float(Kc_tot*conv)
out['L5_K_over_phi0_Gy_cm2']=float(K_tot*conv)
out['_Dz']=[float(zg[i]) for i in range(0,len(zg),20)], [float(Dz[i]) for i in range(0,len(zg),20)]
""")
s=s.replace("print(\"m\",m","#")
s+="""
pickle.dump(out,open('/tmp/work/out.pkl','wb'))
"""
open('/tmp/work/full.py','w').write(s)
import subprocess
r=subprocess.run(['python3','/tmp/work/full.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-3000:], r.stderr[-3000:])
import pickle, json, numpy as np
out=pickle.load(open('/tmp/work/out.pkl','rb'))
for k,v in out.items():
    if isinstance(v,(list,)) and len(v)>0 and not isinstance(v[0],(list,)) : print(k, np.round(np.array(v),5))
    elif isinstance(v,dict): print(k, v)
    else: print(k, v)
