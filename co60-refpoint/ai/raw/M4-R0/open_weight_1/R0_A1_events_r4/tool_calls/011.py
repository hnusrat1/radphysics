import numpy as np, math, random
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

print(muinc(1.1732))