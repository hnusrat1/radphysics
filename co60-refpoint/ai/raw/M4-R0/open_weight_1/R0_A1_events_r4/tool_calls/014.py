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

# Geometry functions
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

# Compton sampling
def sample_compton(E):
    a=E/mec2
    while True:
        mu=random.uniform(-1.0,1.0)
        k=1.0/(1.0+a*(1.0-mu))
        f=k*k*(k+1.0/k+mu*mu-1.0)
        if random.random()<=f/2.0:
            break
    Eout=k*E
    te=E-Eout
    theta=math.acos(mu)
    if abs(theta)<1e-10:
        cose=1.0
    else:
        tan_half=math.tan(theta/2.0)
        cot_phi=(1.0+a)*tan_half
        phi=math.atan2(1.0,cot_phi)
        cose=math.cos(phi)
    return Eout, mu, te, cose

# Rayleigh sampling (coherent, forward peaked)
def sample_rayleigh(E):
    # Sample theta from Gaussian with sigma ~ 0.01 rad
    sigma=0.008
    while True:
        theta=random.gauss(0,sigma)
        if 0<=theta<=math.pi:
            break
    mu=math.cos(theta)
    return E, mu  # no energy loss

# Pair production sampling
def sample_pair(E):
    Eavail=E-2*mec2
    # Energy sharing: uniform
    te=random.uniform(0,Eavail)
    tp=Eavail-te
    # Angles: forward peaked
    sigma=mec2/E
    while True:
        theta=random.gauss(0,sigma)
        if 0<=theta<=math.pi/2:
            break
    cose=math.cos(theta)
    # Positron approximately same angle (rough approximation)
    return te, cose, tp, cose

# Photoelectric sampling
def sample_pe(E):
    te=E  # binding energy negligible
    sigma=0.5
    while True:
        theta=random.gauss(0,sigma)
        if 0<=theta<=math.pi/2:
            break
    cose=math.cos(theta)
    return te, cose

# Rotate direction
def rotate_dir(dx,dy,dz,cos_theta,phi):
    sin_theta=math.sqrt(1-cos_theta*cos_theta)
    # Find perpendicular vector
    if abs(dx)<0.9:
        px,py,pz=1.0,0.0,0.0
    else:
        px,py,pz=0.0,1.0,0.0
    dot=px*dx+py*dy+pz*dz
    px-=dot*dx; py-=dot*dy; pz-=dot*dz
    n=math.sqrt(px*px+py*py+pz*pz)
    if n>0:
        px/=n; py/=n; pz/=n
    # Second perpendicular
    qx=dy*pz-dz*py
    qy=dz*px-dx*pz
    qz=dx*py-dy*px
    # New direction
    dxn=cos_theta*dx+sin_theta*(math.cos(phi)*px+math.sin(phi)*qx)
    dyn=cos_theta*dy+sin_theta*(math.cos(phi)*py+math.sin(phi)*qy)
    dzn=cos_theta*dz+sin_theta*(math.cos(phi)*pz+math.sin(phi)*qz)
    nn=math.sqrt(dxn*dxn+dyn*dyn+dzn*dzn)
    if nn>0:
        return dxn/nn, dyn/nn, dzn/nn
    else:
        return dx,dy,dz

# Simulate one photon history
def simulate_one():
    events=[]
    # Initial energy
    if random.random()<0.5:
        E=1.1732
    else:
        E=1.3325
    dx,dy,dz=sample_dir()
    x,y,z=0.0,0.0,sz
    
    max_steps=100
    step=0
    while step<max_steps:
        step+=1
        mt=mutot(E)
        if mt<=0:
            break
        s=-math.log(random.random())/mt
        xn=x+s*dx
        yn=y+s*dy
        zn=z+s*dz
        
        if not in_phantom(xn,yn,zn):
            break
        
        x,y,z=xn,yn,zn
        
        mu_c=muinc(E)
        mu_p=mupp(E)
        mu_pe=mupe(E)
        mu_r=muray(E)
        mt=mu_c+mu_p+mu_pe+mu_r
        
        r=random.random()*mt
        
        if r<mu_c:
            # Compton
            Eout, cos_scat, te, cose=sample_compton(E)
            if in_scoring(x,y,z):
                events.append({
                    "process":"compton",
                    "e_in_MeV":E,
                    "e_out_MeV":Eout,
                    "cos_scatter":cos_scat,
                    "t_e_MeV":te,
                    "cos_electron":cose,
                    "t_positron_MeV":None
                })
            # Continue tracking scattered photon
            E=Eout
            phi=random.uniform(0,2*math.pi)
            dx,dy,dz=rotate_dir(dx,dy,dz,cos_scat,phi)
            
        elif r<mu_c+mu_p:
            # Pair production
            te, cose, tp, cosp=sample_pair(E)
            if in_scoring(x,y,z):
                events.append({
                    "process":"pair",
                    "e_in_MeV":E,
                    "e_out_MeV":None,
                    "cos_scatter":None,
                    "t_e_MeV":te,
                    "cos_electron":cose,
                    "t_positron_MeV":tp
                })
            break  # photon absorbed
            
        elif r<mu_c+mu_p+mu_pe:
            # Photoelectric
            te, cose=sample_pe(E)
            if in_scoring(x,y,z):
                events.append({
                    "process":"photoelectric",
                    "e_in_MeV":E,
                    "e_out_MeV":None,
                    "cos_scatter":None,
                    "t_e_MeV":te,
                    "cos_electron":cose,
                    "t_positron_MeV":None
                })
            break  # photon absorbed
            
        else:
            # Rayleigh
            Eout, cos_scat=sample_rayleigh(E)
            if in_scoring(x,y,z):
                events.append({
                    "process":"rayleigh",
                    "e_in_MeV":E,
                    "e_out_MeV":Eout,
                    "cos_scatter":cos_scat,
                    "t_e_MeV":0.0,
                    "cos_electron":None,
                    "t_positron_MeV":None
                })
            phi=random.uniform(0,2*math.pi)
            dx,dy,dz=rotate_dir(dx,dy,dz,cos_scat,phi)
            # E unchanged
    
    return events

# Run simulation
print("Starting simulation...")
all_events=[]
nhist=0
target=200
max_hist=500000
while len(all_events)<target and nhist<max_hist:
    nhist+=1
    if nhist%50000==0:
        print(f"  History {nhist}, collected {len(all_events)} events")
    evts=simulate_one()
    all_events.extend(evts)

print(f"Done. Histories: {nhist}, total events: {len(all_events)}")
events_out=all_events[:200]
print(f"Events collected: {len(events_out)}")

# Print summary
procs=[e["process"] for e in events_out]
from collections import Counter
print(Counter(procs))