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
        t=E-2*mec2; 
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
    elif E>=2.0: return 0.00078*(E/2)**(-3)
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
        return math.exp(n.interp(le,lea,lma))

print("Functions defined")

# Check transport
def sample_dir():
    x0=random.uniform(-fh,fh)
    y0=random.uniform(-fh,fh)
    dx=x0; dy=y0; dz=100.0
    n=math.sqrt(dx*dx+dy*dy+dz*dz)
    return dx/n, dy/n, dz/n

# Test: transport from source to phantom and first interaction
def test_transport():
    # Source at (0,0,-100)
    # Direction
    dx,dy,dz = sample_dir()
    # Distance to phantom surface (z=0) along ray
    # z = -100 + t*dz = 0 => t = 100/dz
    t_to_phantom = 100.0/dz  # since dz > 0 for forward direction
    # Entry point
    x0 = dx * t_to_phantom
    y0 = dy * t_to_phantom
    z0 = 0.0
    
    print(f"Entry point: ({x0:.4f}, {y0:.4f}, {z0:.4f})")
    print(f"Distance to phantom: {t_to_phantom:.4f}")
    
    # Now sample interaction inside phantom
    # The photon can interact at a distance s into the phantom
    # Probability of no interaction up to s: exp(-mu*s)
    # Sample s: s = -ln(xi)/mu
    # But we also need to check if the photon exits the phantom before interacting
    
    E = 1.1732  # test
    mu = muinc(E) + mupp(E) + mupe(E) + muray(E)
    print(f"mu_total = {mu:.6f} cm^-1")
    print(f"Mean free path = {1/mu:.4f} cm")
    
    # Sample interaction distance into phantom
    s = -math.log(random.random())/mu
    z_interact = s * dz  # since dz is direction cosine
    # Actually, the distance traveled in the phantom is s, so the z-coordinate increases by s*dz
    z_int = s * dz
    
    print(f"Sampled path length: {s:.4f} cm")
    print(f"Interaction z: {z_int:.4f} cm")
    
    # Check if within phantom
    if z_int <= 30:
        print("Interaction inside phantom")
    else:
        print("Photon exits phantom")

# Run test a few times
for i in range(5):
    print(f"\n--- Test {i} ---")
    test_transport()