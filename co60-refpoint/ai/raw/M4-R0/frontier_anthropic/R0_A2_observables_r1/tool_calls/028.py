
import numpy as np, json
R=json.load(open('/tmp/work/results.json'))
prof={float(k):v for k,v in R['_profile_Kcol'].items()}
zs=np.array(sorted(prof)); Ks=np.array([prof[z] for z in zs])
# extrapolate K to z=0 linearly in log, use fine grid
zg=np.linspace(0,5,5001)
Kg=np.exp(np.interp(zg,zs,np.log(Ks)))
Kg=np.where(zg<zs[0], Ks[0]*np.exp((zg-zs[0])*(np.log(Ks[1]/Ks[0])/(zs[1]-zs[0]))),Kg)
dz=zg[1]-zg[0]
res={}
for lam in [0.05,0.07,0.09,0.12,0.15,0.2,0.3]:
    k=np.exp(-zg/lam)/lam
    D=np.array([np.sum(Kg[:i+1][::-1]*k[:i+1])*dz for i in range(len(zg))])  # D(z)=∫_0^z K(z') k(z-z') dz'
    izm=np.argmax(D[:1001]) # search up to 1 cm
    res[lam]=(zg[izm], 100*D[np.argmin(abs(zg-5))]/D[izm])
    print(lam, zg[izm], round(100*D[np.argmin(abs(zg-5))]/D[izm],2))
