import numpy as np
import math
import random

mec2=0.511
r0=2.8179403262e-13
nepm3=3.3456e23
source_z=-100.0
pz_min=0.0
pz_max=30.0
phalf=15.0
sc_r=1.0
sc_z0=5.0
sc_dz=0.1
field_half=5.0

def mu_inc(E):
    if E<=0: return 0.0
    a=E/mec2; a2=a*a; x=1+2*a; lx=math.log(x)
    t1=(1+a)/a2*(2*(1+a)/x-lx/a)
    t2=lx/(2*a)-(1+3*a)/(x*x)
    return 2*math.pi*r0*r0*(t1+t2)*nepm3

def mu_pp(E):
    if E<=2*mec2: return 0.0
    e_arr=np.array([1.1732,1.3325,2.0,3.0,5.0,10.0])
    mu_arr=np.array([0.006205,0.006286,0.0082,0.0109,0.0126,0.0120])
    if E<=1.1732:
        t=E-2*mec2
        if t<=0: return 0.0
        t0=1.1732-1.022
        return 0.006205*(t/t0)**2
    elif E>=10.0:
        return 0.0120
    else:
        le=math.log(E); lea=np.log(e_arr); lma=np.log(mu_arr)
        return math.exp(np.interp(le,lea,lma))

def mu_pe(E):
    if E<=0: return 0.0
    e_arr=np.array([0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0,1.1732,1.3325,2.0])
    mu_arr=np.array([5.33,0.742,0.0321,0.0164,0.0097,0.0057,0.0026,0.0011,0.001735,0.001524,0.00078])
    if E<=0.01:
        return 5.33*(E/0.01)**(-3.5)
    elif E>=2.0:
        return 0.00078*(E/2.0)**(-3.0)
    else:
        le=math.log(E); lea=np.log(e_arr); lma=np.log(mu_arr)
        return math.exp(np.interp(le,lea,lma))

def mu_ray(E):
    if E<=0: return 0.0
    e_arr=np.array([0.01,0.02,0.05,0.1,0.2,0.5,1.0,1.1732,1.3325,2.0,5.0])
    mu_arr=np.array([0.0008,0.0010,0.0012,0.0016,0.0021,0.0024,0.0025,0.00253,0.00213,0.0012,0.0003])
    if E<=0.01:
        return 0.0008*(E/0.01)**0.3
    elif E>=5.0:
        return 0.0003*(E/5.0)**(-1.5)
    else:
        le=math.log(E); lea=np.log(e_arr); lma=np.log(mu_arr)
        return math.exp(np.interp(le,lea,lma))

def mu_tot(E):
    return mu_inc(E)+mu_pp(E)+mu_pe(E)+mu_ray(E)

def in_phantom(x,y,z):
    return pz_min<=z<=pz_max and -phalf<=x<=phalf and -phalf<=y<=phalf

def in_scoring(x,y,z):
    return (x*x+y*y<=sc_r*sc_r) and abs(z-sc_z0)<=sc_dz

def sample_dir():
    x0=random.uniform(-field_half,field_half)
    y0=random.uniform(-field_half,field_half)
    dx=x0; dy=y0; dz=100.0
    n=math.sqrt(dx*dx+dy*dy+dz*dz)
    return dx/n,dy/n,dz/n

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
        tan_half=math.tan(theta/2)
        cot_phi=(1.0+a)*tan_half
        phi=math.atan2(1.0,cot_phi)
        cose=math.cos(phi)
    return Eout,mu,te,cose

def sample_rayleigh(E):
    sigma=0.008
    while True:
        theta=random.gauss(0,sigma)
        if 0<=theta<=math.pi:
            break
    mu=math.cos(theta)
    return E,mu

def sample_pair(E):
    Eavail=E-2*mec2
    te=random.uniform(0,Eavail)
    tp=Eavail-te
    sigma=mec2/E
    while True:
        theta=random.gauss(0,sigma)
        if 0<=theta<=math.pi/2:
            break
    cose=math.cos(theta)
    cosp=cose
    return te,cose,tp,cosp

def sample_pe(E):
    te=E
    sigma=0.3
    while True:
        theta=random.gauss(0,sigma)
        if 0<=theta<=math.pi/2:
            break
    cose=math.cos(theta)
    return te,cose

def rotate_dir(dx,dy,dz,cos_theta,phi):
    sin_theta=math.sqrt(1-cos_theta*cos_theta)
    if abs(dx)<0.9:
        px,py,pz=1.0,0.0,0.0
    else:
        px,py,pz=0.0,1.0,0.0
    dot=px*dx+py*dy+pz*dz
    px-=dot*dx; py-=dot*dy; pz-=dot*dz
    n=math.sqrt(px*px+py*py+pz*pz)
    if n>0:
        px/=n; py/=n; pz/=n
    qx=dy*pz-dz*py
    qy=dz*px-dx*pz
    qz=dx*py-dy*px
    dxn=cos_theta*dx+sin_theta*(math.cos(phi)*px+math.sin(phi)*qx)
    dyn=cos_theta*dy+sin_theta*(math.cos(phi)*py+math.sin(phi)*qy)
    dzn=cos_theta*dz+sin_theta*(math.cos(phi)*pz+math.sin(phi)*qz)
    nn=math.sqrt(dxn*dxn+dyn*dyn+dzn*dzn)
    if nn>0:
        return dxn/nn,dyn/nn,dzn/nn
    return dx,dy,dz

def simulate_history():
    events=[]
    if random.random()<0.5:
        E=1.1732
    else:
        E=1.3325
    dx,dy,dz=sample_dir()
    dist=(pz_min-source_z)/dz
    x=dx*dist
    y=dy*dist
    z=pz_min
    
    for _ in range(200):
        mt=mu_tot(E)
        if mt<=0: break
        s=-math.log(random.random())/mt
        xn=x+s*dx
        yn=y+s*dy
        zn=z+s*dz
        if not in_phantom(xn,yn,zn):
            break
        x,y,z=xn,yn,zn
        
        mc=mu_inc(E); mp=mu_pp(E); mpe=mu_pe(E); mr=mu_ray(E)
        mt=mc+mp+mpe+mr
        r=random.random()*mt
        
        if r<mc:
            Eout,cos_scat,te,cose=sample_compton(E)
            if in_scoring(x,y,z):
                events.append({"process":"compton","e_in_MeV":E,"e_out_MeV":Eout,"cos_scatter":cos_scat,"t_e_MeV":te,"cos_electron":cose,"t_positron_MeV":None})
            E=Eout
            phi=random.uniform(0,2*math.pi)
            dx,dy,dz=rotate_dir(dx,dy,dz,cos_scat,phi)
        elif r<mc+mp:
            te,cose,tp,cosp=sample_pair(E)
            if in_scoring(x,y,z):
                events.append({"process":"pair","e_in_MeV":E,"e_out_MeV":None,"cos_scatter":None,"t_e_MeV":te,"cos_electron":cose,"t_positron_MeV":tp})
            break
        elif r<mc+mp+mpe:
            te,cose=sample_pe(E)
            if in_scoring(x,y,z):
                events.append({"process":"photoelectric","e_in_MeV":E,"e_out_MeV":None,"cos_scatter":None,"t_e_MeV":te,"cos_electron":cose,"t_positron_MeV":None})
            break
        else:
            Eout,cos_scat=sample_rayleigh(E)
            if in_scoring(x,y,z):
                events.append({"process":"rayleigh","e_in_MeV":E,"e_out_MeV":Eout,"cos_scatter":cos_scat,"t_e_MeV":0.0,"cos_electron":None,"t_positron_MeV":None})
            phi=random.uniform(0,2*math.pi)
            dx,dy,dz=rotate_dir(dx,dy,dz,cos_scat,phi)
    return events

print("Starting simulation...")
random.seed(1234)
all_ev=[]
n_hist=0
target=200
max_hist=5000000
while len(all_ev)<target and n_hist<max_hist:
    n_hist+=1
    if n_hist%200000==0:
        print(f"  History {n_hist}, events {len(all_ev)}")
    evts=simulate_history()
    all_ev.extend(evts)

print(f"Done. Histories: {n_hist}, events: {len(all_ev)}")
ev_final=all_ev[:200]
print(f"Events: {len(ev_final)}")
from collections import Counter
print(Counter([e["process"] for e in ev_final]))