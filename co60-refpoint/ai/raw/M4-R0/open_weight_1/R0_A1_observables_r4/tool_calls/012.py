
import numpy as np
# test score_cylinder
def score_cylinder(x,y,z,ux,uy,uz,s):
    zlo=4.9; zhi=5.1; rad=1.0
    tz1=(zlo-z)/uz; tz2=(zhi-z)/uz
    tza=np.minimum(tz1,tz2); tzb=np.maximum(tz1,tz2)
    tza=np.clip(tza,0.0,s); tzb=np.clip(tzb,0.0,s)
    uz0 = np.abs(uz)<1e-9
    tza=np.where(uz0, 0.0, tza); tzb=np.where(uz0, s, tzb)
    tza=np.where(uz0 & ((z<zlo)|(z>zhi)), 0.0, tza)
    tzb=np.where(uz0 & ((z<zlo)|(z>zhi)), 0.0, tzb)
    a=ux*ux+uy*uy
    b=2.0*(x*ux+y*uy)
    c=x*x+y*y-rad*rad
    disc=b*b-4*a*c
    a0 = np.abs(a)<1e-9
    sq=np.sqrt(np.clip(disc,0,None))
    den=2.0*a+1e-30
    t1=(-b-sq)/den; t2=(-b+sq)/den
    neg = disc<0
    tlo=np.where(neg, 0.0, np.minimum(t1,t2))
    thi=np.where(neg, np.where(c<0, s, 0.0), np.maximum(t1,t2))
    start_in = c<0
    tlo=np.where(start_in & ~neg & ~a0, 0.0, tlo)
    thi=np.where(start_in & ~neg & ~a0, np.maximum(t1,t2), thi)
    tlo=np.where(a0, 0.0, tlo)
    thi=np.where(a0, np.where(c<0,s,0.0), thi)
    tlo=np.maximum(tlo, tza); thi=np.minimum(thi, tzb)
    return np.maximum(thi-tlo,0.0)

# case 1: vertical photon on axis, from z=0, s=6
print(score_cylinder(np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([1.0]),np.array([6.0])))
# case 2: vertical photon at x=2 (outside radius), z=0, s=6 -> should be 0
print(score_cylinder(np.array([2.0]),np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([1.0]),np.array([6.0])))
# case 3: photon at x=0.5, vertical, z=0..6 -> 0.2
print(score_cylinder(np.array([0.5]),np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([0.0]),np.array([1.0]),np.array([6.0])))
# case 4: photon entering from z=4.0 moving +z with uz=1, s=2 -> crosses 4.9-5.1: 0.2
print(score_cylinder(np.array([0.0]),np.array([0.0]),np.array([4.0]),np.array([0.0]),np.array([0.0]),np.array([1.0]),np.array([2.0])))
# case 5: photon at z=4.5 moving to z=6: s=1.5
print(score_cylinder(np.array([0.0]),np.array([0.0]),np.array([4.5]),np.array([0.0]),np.array([0.0]),np.array([1.0]),np.array([1.5])))
