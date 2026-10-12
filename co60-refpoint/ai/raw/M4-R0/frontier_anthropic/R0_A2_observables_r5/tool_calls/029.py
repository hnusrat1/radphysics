
pdd2 = r'''
import sys, pickle, json; sys.path.insert(0,'/tmp/work')
import numpy as np
import mcana, mcmod, ana as A
from mcana import *
from mcmod import *

def depth_K(t2, nh2, LATv=1.0):
    zc = np.linspace(0,10,101); zm=0.5*(zc[:-1]+zc[1:])
    Vb = (2*LATv)**2*0.1
    Ks = t2['zb_kcol']/nh2/Vb
    Pphi0 = A.primary_fluence_per_phi0()
    Kp = np.zeros(100)
    for E in LINES:
        mu=mu_tot(E); mue=mu_en_all(E)
        Kp += 0.5*Phi0_src*np.exp(-mu*zm)*(105.0/(100.0+zm))**2*E*mue
    return zm, Ks, Kp

def kernel(C_nc, cF, zbin=0.1, xmin=-1.5, xmax=6.0, ns=3000):
    xg = np.arange(xmin, xmax+zbin/2, zbin)
    nb=len(xg)-1
    kern=np.zeros(nb)
    cen=np.linspace(-0.95,0.95,20)   # finer angular grid (uniform in cos over bins)
    # distribute C_nc (10 cos bins over [-1,1]) onto 20 subcells: use bin-centre cos of the 10 bins
    cb = np.array([-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9])
    for i,T in enumerate(cF):
        if T<0.0105: continue
        R=CSDA(T)
        s=np.linspace(0,R,ns+1); sm=0.5*(s[1:]+s[:-1]); ds=s[1]-s[0]
        Rrem=np.maximum(R-sm,1e-12)
        Eloc=np.exp(np.interp(np.log(Rrem),np.log(RANGE),np.log(TE)))
        dep=S_col(Eloc)*ds
        for j,c in enumerate(cb):
            w=C_nc[i,j]
            if w<=0: continue
            x=c*sm
            idx=np.floor((x-xmin)/zbin).astype(int)
            ok=(idx>=0)&(idx<nb)
            np.add.at(kern, idx[ok], w*dep[ok])
    return xg, kern

def run(t1,t2,ptal,nh1,nh2):
    cF=np.sqrt(FINE[:-1]*FINE[1:])
    Cnc = (t1['q_fine_all_cos']+t2['q_fine_all_cos'])/(nh1+nh2) + ptal['q_fine_all_cos']
    xg,kern = kernel(Cnc, cF)
    kn = kern/kern.sum()
    zm, Ks, Kp = depth_K(t2, nh2)
    Ktot = (Ks+Kp)/Phi0_src
    # convolution: D(z_i)=sum_j Ktot(z_j) kn(z_i-z_j)
    nz=len(zm)
    D=np.zeros(nz)
    for i in range(nz):
        for j in range(nz):
            k=int(np.round((zm[i]-zm[j]-(xg[0]+0.05))/0.1))
            if 0<=k<len(kn): D[i]+=Ktot[j]*kn[k]
    res=dict(zm=zm.tolist(), Ktot=Ktot.tolist(), D=D.tolist(), kn=kn.tolist(), xg=xg.tolist())
    return res

if __name__=='__main__':
    t1=pickle.load(open('/tmp/work/tal_run1.pkl','rb'))
    t2=pickle.load(open('/tmp/work/tal_run2_lat1.pkl','rb'))
    ptal=pickle.load(open('/tmp/work/tal_all.pkl','rb'))[1]
    res=run(t1,t2,ptal,t1['nh'],t2['nh'])
    pickle.dump(res,open('/tmp/work/pdd_res.pkl','wb'))
    zm=np.array(res['zm']); D=np.array(res['D']); Kt=np.array(res['Ktot'])
    imax=np.argmax(D)
    print("dmax (bin centre)", zm[imax], "D max rel", D[imax])
    i5=np.argmin(np.abs(zm-5.0))
    print("PDD5", 100*D[i5]/D[imax], " Ktot ratio K(5)/Kmax", Kt[i5]/Kt.max())
    print("kernel mean depth", np.sum(np.array(res['kn'])*(np.array(res['xg'][:-1])+0.05)))
    print("D first bins/Dmax", np.round(D[:15]/D.max(),3).tolist())
    print("D rel at 0.5..1:", np.round(D[4:12]/D.max(),3).tolist())
'''
open('/tmp/work/pdd2.py','w').write(pdd2)
import subprocess
r=subprocess.run(['python3','/tmp/work/pdd2.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-3000:], r.stderr[-3000:])
