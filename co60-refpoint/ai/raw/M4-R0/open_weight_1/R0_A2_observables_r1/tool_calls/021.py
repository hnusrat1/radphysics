import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
slab_src=A[11]; Ntot=int(A[12])
PHI0=1/10.5**2
s = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=s[:,0]; Scol=s[:,1]; Rcsda=s[:,4]
def interp(x,xp,yp): return np.interp(x,xp,yp)
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
emid=(el_edges[:-1]+el_edges[1:])/2
cos_mid=np.linspace(-0.9,0.9,10)
V_slab=math.pi*0.2
Rinv_E=np.linspace(0, interp(1.4,Ee,Rcsda), 4000)
Rinv_val=np.interp(Rinv_E, Rcsda, Ee)

# test single source kernel: n_A=1, E0=0.5, mu=0.9
E0=0.5; mu=0.9; R0=interp(E0,Ee,Rcsda)
sp=np.linspace(0,R0,20000); rem=R0-sp; Ea=np.interp(rem,Rinv_E,Rinv_val); Sc=interp(Ea,Ee,Scol)
Dz = Sc/mu
integral = np.trapezoid(Dz, sp)   # int D dz where dz=mu ds => int Dz dz = int Dz mu ds = int Sc ds
print('int Sc ds (should = E0=0.5):', np.trapezoid(Sc,sp))
print('int D(z) dz = int Sc ds:', integral)

# now compute the total depth-integral of my earlier computed D for the full model and compare with source energy
dz=0.02; zg=np.arange(0,30+dz*0.5,dz); NZ=len(zg)
Dtot=np.zeros(NZ)
for k in range(150):
    z0=0.1+0.2*k; cnt=slab_src[k]
    if cnt.sum()==0: continue
    for b in range(25):
        E0b=emid[b]; R0=interp(E0b,Ee,Rcsda); 
        if R0<=0: continue
        for c in range(10):
            mu0=cos_mid[c]; n=cnt[b,c]
            if n<=0: continue
            mu_use=mu0 if abs(mu0)>=0.05 else (0.05 if mu0>=0 else -0.05)
            sp=np.linspace(0,R0,600); rem=R0-sp
            Ea=np.interp(rem,Rinv_E,Rinv_val); Sc=interp(Ea,Ee,Scol)
            w = n/Ntot/PHI0/V_slab*0.2
            zpos=z0+mu_use*sp; idx=np.clip((zpos/dz).astype(int),0,NZ-1)
            valid=(zpos>=0)&(zpos<=30)
            np.add.at(Dtot, idx[valid], (w*Sc/abs(mu_use))[valid])
intD = np.trapezoid(Dtot, zg)
# total source energy per unit volume per Phi0 integrated over depth:
src_energy_sum = 0.0
for k in range(150):
    cnt=slab_src[k].sum(1)  # (25,)
    src_energy_sum += (cnt*emid).sum()/Ntot/PHI0/V_slab*0.2
print('intD dz =', intD, ' total source energy (per unit Phi0):', src_energy_sum)
print('D(5):', Dtot[int(5/dz)], 'Dmax', Dtot.max(), 'at', zg[int(np.argmax(Dtot))])