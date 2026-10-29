import numpy as np
import math
import random

mec2=0.511
r0=2.8179403262e-13
nepm3=3.3456e23
sz=-100.0
pz0=0.0
pz1=30.0
ph=15.0
sr=1.0
sz0=5.0
sdz=0.1
fh=5.0

def muinc(E):
    if E<=0: return 0.0
    a=E/mec2; a2=a*a; x=1+2*a; lx=math.log(x)
    t1=(1+a)/a2*(2*(1+a)/x-lx/a)
    t2=lx/(2*a)-(1+3*a)/(x*x)
    return 2*math.pi*r0*r0*(t1+t2)*nepm3

def mupp(E):
    if E<=2*mec2: return 0.0
    e=np.array([1.1732,1.3325,2.0,3.0,5.0,10.0])
    m=np.array([0.006205,0.006286,0.0082,0.0109,0.0126,0.0120])
    if E<=1.1732:
        t=E-2*mec2
        if t<=0: return 0.0
        t0=1.1732-1.022
        return 0.006205*(t/t0)**2
    elif E>=10.0: return 0.0120
    else:
        le=math.log(E); lea=np.log(e); lma=np.log(m)
        return math.exp(np.interp(le,lea,lma))

def mupe(E):
    if E<=0: return 0.0
    e=np.array([0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0,1.1732,1.3325,2.0])
    m=np.array([5.33,0.742,0.0321,0.0164,0.0097,0.0057,0.0026,0.0011,0.001735,0.001524,0.00078])
    if E<=0.01: return 5.33*(E/0.01)**(-3.5)
    elif E>=2.0: return 0.00078*(E/2.0)**(-3.0)
    else:
        le=math.log(E); lea=np.log(e); lma=np.log(m)
        return math.exp(np.interp(le,lea,lma))

def muray(E):
    if E<=0: return 0.0
    e=np.array([0.01,0.02,0.05,0.1,0.2,0.5,1.0,1.1732,1.3325,2.0,5.0])
    m=np.array([0.0008,0.0010,0.0012,0.0016,0.0021,0.0024,0.0025,0.00253,0.00213,0.0012,0.0003])
    if E<=0.01: return 0.0008*(E/0.01)**0.3
    elif E>=5.0: return 0.0003*(E/5.0)**(-1.5)
    else:
        le=math.log(E); lea=np.log(e); lma=np.log(m)
        return math.exp(np.interp(le,lea,lma))