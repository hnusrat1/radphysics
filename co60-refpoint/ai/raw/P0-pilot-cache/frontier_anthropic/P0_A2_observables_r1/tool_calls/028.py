
import sys, pickle; sys.path.insert(0,'/tmp/work')
import numpy as np, xs
tot=pickle.load(open('/tmp/work/tot.pkl','rb')); res=pickle.load(open('/tmp/work/eres1.pkl','rb'))
N=tot['N']; PHI=1/(0.0099751*110**2); VC=np.pi*0.2
Vslab=np.pi*9*0.25
Qs=tot['Qslab']/(N*Vslab*PHI)   # MeV/cm3 per Phi0 per 0.25cm slab, centers 0.125+0.25j
print(np.round(Qs[:16],3), Qs[[20,30,40]])
# fine grid 0.02 cm
dz=0.02
zg=np.arange(0,12.0,dz)+dz/2
Qfine=np.array([Qs[int(z//0.25)] if int(z//0.25)<len(Qs) else 0 for z in zg])
kern=res['kern']/res['T0s']          # fraction per 0.02 bin, kernel from ZB
print('kernel total fraction', kern.sum())
# kernel index j from -200..199 mapping to delta=j*0.02
pj=np.zeros(401); 
pj[1:]=0
kern_d=np.zeros(401)  # index j+200
kern_d[:len(kern)]=kern[:401] if len(kern)>=401 else 0
# kern has len 400 (edges -4..4 step .02 => 400 bins). centers -3.99..3.99; map bin k -> j=k-200 (+0.5)
kd=np.zeros(401)
for k in range(400):
    j=k-200   # delta index; center (k-200+0.5)*0.02 ~ j*0.02
    kd[j+200 if 0<=j+200<=400 else 0]+=0
kd=np.zeros(801)  # offsets -400..400 in 0.02 steps? use direct mapping below
offs=np.arange(-200,200)  # bins index m-i
pk=kern  # 400 entries, centers at (k-200+0.5)*dz
# discrete convolution: D_m = sum_i Q_i * p(z_m - z_i), with z_m-z_i = (m-i)*dz ~ center (k-200+0.5)dz => k = m-i+200
M=len(zg); D=np.zeros(M)
for m in range(M):
    i=np.arange(M)
    kidx=(m-i)+200
    ok=(kidx>=0)&(kidx<400)
    D[m]=np.sum(Qfine[ok]*pk[kidx[ok]])
zi=zg
for zt in [0.5,1,1.5,2,3,4,5,6,8,10]:
    m=int(zt/dz); print(zt, D[m], Qfine[m])
imax=np.argmax(D); print('dmax',zg[imax], D[imax])
