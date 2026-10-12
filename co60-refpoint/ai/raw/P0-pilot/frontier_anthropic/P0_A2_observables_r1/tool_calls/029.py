
code_pdd = r'''
import sys, pickle
sys.path.insert(0,'/tmp/work')
import numpy as np
from phys import mu_all
D=pickle.load(open('/tmp/work/photon_tallies.pkl','rb'))
tot=D['tot']
K=pickle.load(open('/tmp/work/kern.pkl','rb'))
hist=K['hist']; ZB=K['ZB']
import deposit
dep2=deposit.transfer_coeffs(np.array([2.0]))
ZD=np.arange(0.25,20.0001,0.25)
muT=float(mu_all(np.array([2.0]))[4][0])
Kd=tot['Kd']
Gz=1.602e-10*(np.exp(-muT*ZD)*float(dep2['col'][0])+Kd)     # Kcol per unit volume per Phi0 (Gy cm2/cm) generation at depth ZD
E_tot=hist.sum()
# kernel: energy fraction in bin of width 0.05 cm, centred on Delta=m*0.05, m index: bin b=m+120
dz=0.05
zgrid=np.arange(0,20.0+1e-9,dz)+0.0
G=np.interp(zgrid,np.concatenate([[0.0],ZD]),np.concatenate([[Gz[0]*0+Gz[0]],Gz]))
# optional: G at z=0 ~ G(0.25)
Kfrac=hist/E_tot
def kern(m):
    b=m+120
    return Kfrac[b] if 0<=b<len(Kfrac) else 0.0
ms=np.arange(-120,161)
kvals=np.array([kern(m) for m in ms])
Dz=np.zeros(len(zgrid))
for i in range(len(zgrid)):
    # sum over source positions j: Delta = (i-j)*dz
    j=np.arange(len(zgrid))
    mm=i-j
    valid=(mm>=-120)&(mm<=160)
    Dz[i]=np.sum(G[j[valid]]*kvals[mm[valid]+120])*dz/dz  # G per unit volume times fraction per bin
zz=zgrid
# restrict to depths within phantom and where G defined (<=20 cm); beyond 19 cm the kernel tail truncated -> only use up to 15 cm
mask=zz<=15.0
imax=np.argmax(Dz[mask])
dmax=zz[mask][imax]
d5=np.argmin(np.abs(zz-5.0))
PDD5=100*Dz[d5]/Dz[mask][imax]
Kcol_check=Gz[np.argmin(np.abs(ZD-10))]
print("dmax",dmax,"D(dmax)",Dz[mask][imax],"D(5)",Dz[d5],"PDD5",PDD5,"D(10)",Dz[np.argmin(np.abs(zz-10))], "Kcol(10)",Kcol_check)
print("PDD profile (depth,PDD%):",[(round(zz[i],2), round(100*Dz[i]/Dz[mask][imax],2)) for i in range(0,len(zz),20) if zz[i]<=15])
out=dict(dmax=float(dmax),PDD5=float(PDD5),D_dmax=float(Dz[mask][imax]),D5=float(Dz[d5]),D10=float(Dz[np.argmin(np.abs(zz-10))]))
pickle.dump(dict(out=out,zz=zz,Dz=Dz),open('/tmp/work/pdd.pkl','wb'))
print(out)
'''
open('/tmp/work/pdd.py','w').write(code_pdd)
import subprocess
r=subprocess.run(['python3','/tmp/work/pdd.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-5000:], r.stderr[-3000:])
