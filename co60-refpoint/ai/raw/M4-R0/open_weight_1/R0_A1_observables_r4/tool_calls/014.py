
import numpy as np
def score_cylinder(x,y,z,ux,uy,uz,s):
    zlo=4.9; zhi=5.1; rad=1.0
    tz1=(zlo-z)/uz; tz2=(zhi-z)/uz
    tza=np.minimum(tz1,tz2); tzb=np.maximum(tz1,tz2)
    tza=np.clip(tza,0.0,s); tzb=np.clip(tzb,0.0,s)
    uz0=np.abs(uz)<1e-9
    tza=np.where(uz0,0.0,tza); tzb=np.where(uz0,s,tzb)
    tza=np.where(uz0&((z<zlo)|(z>zhi)),0.0,tza); tzb=np.where(uz0&((z<zlo)|(z>zhi)),0.0,tzb)
    a=ux*ux+uy*uy; b=2*(x*ux+y*uy); c=x*x+y*y-rad*rad
    disc=b*b-4*a*c; a0=np.abs(a)<1e-9
    sq=np.sqrt(np.clip(disc,0,None)); den=2*a+1e-30
    t1=(-b-sq)/den; t2=(-b+sq)/den; neg=disc<0
    tlo=np.where(neg,0.0,np.minimum(t1,t2)); thi=np.where(neg,np.where(c<0,s,0.0),np.maximum(t1,t2))
    start_in=c<0
    tlo=np.where(start_in&~neg&~a0,0.0,tlo); thi=np.where(start_in&~neg&~a0,np.maximum(t1,t2),thi)
    tlo=np.where(a0,0.0,tlo); thi=np.where(a0,np.where(c<0,s,0.0),thi)
    tlo=np.maximum(tlo,tza); thi=np.minimum(thi,tzb)
    return np.maximum(thi-tlo,0.0)

# off-axis photon: x0=4, y0=0, source at z=-100: direction
x0=4.0; y0=0.0
d=np.sqrt(x0*x0+y0*y0+10000)
ux=x0/d; uy=y0/d; uz=100.0/d
print("dir",ux,uy,uz)
print("at z=5, x=", ux/uz*5)  # slope x/z = ux/uz
print(score_cylinder(np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([ux]),np.array([uy]),np.array([uz]),np.array([6.0])))

# test a photon at x=3, z=0, vertical
print(score_cylinder(np.array([3.0]),np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([1.0]),np.array([6.0])))
