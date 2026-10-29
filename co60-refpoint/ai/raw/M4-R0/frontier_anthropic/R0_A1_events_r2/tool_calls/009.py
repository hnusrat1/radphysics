
test=r'''
import numpy as np, sim
from sim import *
def chord(pos,d,L):
    # chord length within cylinder for segment pos + t d, t in [0,L]
    px,py,pz=pos[:,0],pos[:,1],pos[:,2]; dx,dy,dz=d[:,0],d[:,1],d[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=np.where(np.abs(dz)>1e-12,(4.9-pz)/dz,np.where((pz>=4.9)&(pz<=5.1),-np.inf,np.inf))
        t2=np.where(np.abs(dz)>1e-12,(5.1-pz)/dz,np.where((pz>=4.9)&(pz<=5.1),np.inf,-np.inf))
    tz1=np.minimum(t1,t2); tz2=np.maximum(t1,t2)
    a=dx*dx+dy*dy; b=2*(px*dx+py*dy); c=px*px+py*py-1
    disc=b*b-4*a*c
    with np.errstate(divide='ignore',invalid='ignore'):
        sq=np.sqrt(np.maximum(disc,0))
        tr1=np.where(a>1e-14,(-b-sq)/(2*a),-np.inf); tr2=np.where(a>1e-14,(-b+sq)/(2*a),np.inf)
    tr1=np.where(disc>=0,tr1,np.inf); tr2=np.where(disc>=0,tr2,-np.inf)
    lo=np.maximum.reduce([tz1,tr1,np.zeros_like(tz1)]); hi=np.minimum.reduce([tz2,tr2,L])
    return np.maximum(hi-lo,0)
'''
open('chordlib.py','w').write(test)
# modified run with track-length estimator
src=open('sim.py').read()
src2=src.replace("        while len(E)>0:\n            mu=mu_tot(E)\n            s=-np.log(rng.random(len(E)))/mu\n            dbox=box_exit(pos,dr)\n            inside=s<dbox",
"        while len(E)>0:\n            mu=mu_tot(E)\n            s=-np.log(rng.random(len(E)))/mu\n            dbox=box_exit(pos,dr)\n            inside=s<dbox\n            TL.append(np.sum(mu_tot(E)*chord(pos,dr,np.minimum(s,dbox))))")
src2=src2.replace("def run(Ns, chunk=100000):\n    cands=[]","TL=[]\ndef run(Ns, chunk=100000):\n    cands=[]")
src2=src2.replace("from phys import *","from phys import *\nfrom chordlib import chord")
src2=src2.split('if __name__')[0]
open('simtl.py','w').write(src2)
test2=r'''
import numpy as np, simtl
Ns=400000
C=simtl.run(Ns)
# primary chord tally is in candidates; here TL holds scattered+primary segment tallies (segments after first photon start)
tl=np.array(simtl.TL)
print('TL total per source photon (all segments incl primary):', tl.sum()/Ns)
for ty in [0,1,2,3]:
    m=C[:,0]==ty
    print(ty, C[m,1].sum()/Ns)
'''
open('t3.py','w').write(test2)
r=subprocess.run(['python','t3.py'],capture_output=True,text=True,cwd='.'); print(r.stdout,r.stderr[-2000:])
