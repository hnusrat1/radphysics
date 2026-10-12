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

def mutot(E):
    return muinc(E)+mupp(E)+mupe(E)+muray(E)

def in_phantom(x,y,z):
    return pz0<=z<=pz1 and -ph<=x<=ph and -ph<=y<=ph

def in_scoring(x,y,z):
    return (x*x+y*y)<=sr*sr and abs(z-sz0)<=sdz

def sample_dir():
    x0=random.uniform(-fh,fh)
    y0=random.uniform(-fh,fh)
    dx=x0
    dy=y0
    dz=100.0
    n=math.sqrt(dx*dx+dy*dy+dz*dz)
    return dx/n, dy/n, dz/n

# Test sampling directions and positions at z=5
print("Testing geometry...")
# Sample many directions and check where they hit z=5
for i in range(1000):
    dx,dy,dz=sample_dir()
    # Parametric: (x,y,z) = (0,0,sz) + t*(dx,dy,dz)
    # At z=5, t = (5 - sz)/dz = (5+100)/dz = 105/dz
    if abs(dz)>0:
        t=105.0/dz
        x=dx*t
        y=dy*t
        z=dz*t+sz
        if i<5:
            print(f"  dir=({dx:.4f},{dy:.4f},{dz:.4f}), at z=5: x={x:.4f}, y={y:.4f}, r={math.sqrt(x*x+y*y):.4f}")
        if abs(z-5.0)<0.01 and math.sqrt(x*x+y*y)<sr:
            print(f"  Hit scoring! x={x:.4f}, y={y:.4f}")

# Now let's check how many interactions occur at around z=5
# Let's simulate a few histories and track interaction positions
ncheck=10000
count_in_phantom=0
count_near=0
for i in range(ncheck):
    if random.random()<0.5:
        E=1.1732
    else:
        E=1.3325
    dx,dy,dz=sample_dir()
    x,y,z=0.0,0.0,sz
    mt=mutot(E)
    if mt<=0: continue
    s=-math.log(random.random())/mt
    xn=x+s*dx
    yn=y+s*dy
    zn=z+s*dz
    if in_phantom(xn,yn,zn):
        count_in_phantom+=1
        if abs(zn-5.0)<0.5:
            count_near+=1
            if i<10:
                print(f"  Interaction near 5 cm: z={zn:.4f}, x={xn:.4f}, y={yn:.4f}, r={math.sqrt(xn*xn+yn*yn):.4f}")

print(f"Total first interactions in phantom: {count_in_phantom}/{ncheck}")
print(f"Interactions near 5 cm depth (|z-5|<0.5): {count_near}")