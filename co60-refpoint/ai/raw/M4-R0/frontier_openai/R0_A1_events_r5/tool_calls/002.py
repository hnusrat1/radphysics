import numpy as np, math, time
rng=np.random.default_rng(77319)
# reuse mus and sigma_kn

def unit(v): return v/np.linalg.norm(v)
def scatter_dir(u, c, phi):
    # construct perpendicular unit vector
    if abs(u[2]) < 0.9:
        a=np.array([-u[1],u[0],0.0]); a/=np.linalg.norm(a)
    else:
        a=np.array([1.0,0.0,0.0]); a-=np.dot(a,u)*u; a/=np.linalg.norm(a)
    b=np.cross(u,a)
    s=math.sqrt(max(0.0,1-c*c))
    return c*u+s*(math.cos(phi)*a+math.sin(phi)*b)

def sample_kn(E):
    eps=E/me
    while True:
        c=2*rng.random()-1
        k=1/(1+eps*(1-c))
        val=k*k*(k+1/k-(1-c*c))
        if rng.random()*2 <= val:
            return c

def sample_rayleigh_cos():
    # Thomson angular approximation
    while True:
        c=2*rng.random()-1
        if rng.random() < (1+c*c)/2:
            return c

def add_event(ev, nseen, reservoir):
    nseen += 1
    if nseen <= 200: reservoir.append(ev)
    else:
        j=rng.integers(nseen)
        if j<200: reservoir[j]=ev
    return nseen

N=900_000
reservoir=[]; nseen=0; counts=Counter(); t0=time.time()
for ih in range(N):
    E=1.1732 if rng.random()<0.5 else 1.3325
    # Uniform solid angle over 10x10 cm field at z=0, source 100 cm away.
    while True:
        x=10*rng.random()-5; y=10*rng.random()-5
        w=(100/math.sqrt(10000+x*x+y*y))**3
        if rng.random() <= w: break
    pos=np.array([x,y,0.0]); u=unit(np.array([x,y,100.0]))
    stack=[(pos,u,E)]
    while stack:
        pos,u,E=stack.pop()
        # Transport each photon until exit or termination.
        for _ in range(100):
            muc,mup,mur,mupair=mus(E)
            mut=muc+mup+mur+mupair
            if mut<=0: break
            # Distance to nearest phantom boundary
            dists=[]
            for ax in range(3):
                if u[ax]>1e-14: dists.append(([15,15,30][ax]-pos[ax])/u[ax] if ax<2 else (30-pos[ax])/u[ax])
                elif u[ax]<-1e-14: dists.append(([-15,-15,0][ax]-pos[ax])/u[ax])
            db=min(dists) if dists else 0
            step=-math.log(max(rng.random(),1e-300))/mut
            if step>=db: break
            pos=pos+step*u
            q=rng.random()*mut
            if q<muc: proc='compton'
            elif q<muc+mup: proc='photoelectric'
            elif q<muc+mup+mur: proc='rayleigh'
            else: proc='pair'
            scored=(pos[0]*pos[0]+pos[1]*pos[1]<=1 and 4.9<=pos[2]<=5.1)
            if proc=='compton':
                c=sample_kn(E); phi=2*math.pi*rng.random()
                Eout=E/(1+(E/me)*(1-c)); te=E-Eout
                p_e=math.sqrt(max(E*E+Eout*Eout-2*E*Eout*c,0))
                ce=(E-Eout*c)/p_e if p_e>0 else 1.0
                if scored:
                    ev={'process':'compton','e_in_MeV':E,'e_out_MeV':Eout,'cos_scatter':c,'t_e_MeV':te,'cos_electron':max(-1,min(1,ce)),'t_positron_MeV':None}
                    nseen=add_event(ev,nseen,reservoir); counts[proc]+=1
                u=unit(scatter_dir(u,c,phi)); E=Eout
                if E<0.005: break
            elif proc=='photoelectric':
                if scored:
                    ev={'process':'photoelectric','e_in_MeV':E,'e_out_MeV':None,'cos_scatter':None,'t_e_MeV':max(0,E-0.00001),'cos_electron':1.0,'t_positron_MeV':None}
                    nseen=add_event(ev,nseen,reservoir); counts[proc]+=1
                break
            elif proc=='rayleigh':
                c=sample_rayleigh_cos(); phi=2*math.pi*rng.random()
                if scored:
                    ev={'process':'rayleigh','e_in_MeV':E,'e_out_MeV':E,'cos_scatter':c,'t_e_MeV':0.0,'cos_electron':None,'t_positron_MeV':None}
                    nseen=add_event(ev,nseen,reservoir); counts[proc]+=1
                u=unit(scatter_dir(u,c,phi))
            else:
                excess=E-2*me
                frac=rng.random()
                te=excess*frac; tp=excess-te
                ce=math.sqrt(max(0,1-(me/max(te+me,me))**2))*0.9+0.1 # approximate forward direction
                if scored:
                    ev={'process':'pair','e_in_MeV':E,'e_out_MeV':None,'cos_scatter':None,'t_e_MeV':te,'cos_electron':min(1.0,ce),'t_positron_MeV':tp}
                    nseen=add_event(ev,nseen,reservoir); counts[proc]+=1
                # local annihilation photons from a stopped positron
                if E>=2*me:
                    for _ann in range(2):
                        z=2*rng.random()-1; ph=2*math.pi*rng.random(); rr=math.sqrt(1-z*z)
                        stack.append((pos.copy(),np.array([rr*math.cos(ph),rr*math.sin(ph),z]),me))
                break
            if nseen>=1000000: break
        if nseen>=1000000: break
    if ih%200000==0 and ih>0: print(ih,nseen,time.time()-t0)
print('done N',N,'scored',nseen,'time',time.time()-t0,'counts',counts,'reservoir',len(reservoir))