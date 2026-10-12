import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
slab_src=A[11]; Ntot=int(A[12])
PHI0=1/10.5**2
s = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=s[:,0]; Scol=s[:,1]; Rcsda=s[:,4]
def interp(x,xp,yp): return np.interp(x,xp,yp)
V_slab=math.pi*0.2
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
emid=(el_edges[:-1]+el_edges[1:])/2
cos_mid=np.linspace(-0.9,0.9,10)
dz=0.02; zg=np.arange(0,30+dz*0.5,dz); NZ=len(zg)
D=np.zeros(NZ)
# precompute range inverse table on dense grid
Rinv_E=np.linspace(0, interp(1.4,Ee,Rcsda), 2000)
Rinv_val=np.interp(Rinv_E, Rcsda, Ee)
for k in range(150):
    z0=0.1+0.2*k
    cnt=slab_src[k]
    if cnt.sum()==0: continue
    for b in range(25):
        E0=emid[b]; R0=interp(E0,Ee,Rcsda)
        if R0<=0: continue
        for c in range(10):
            mu0=cos_mid[c]
            n=cnt[b,c]
            if n<=0: continue
            mu_use=mu0 if abs(mu0)>=0.05 else (0.05 if mu0>=0 else -0.05)
            if abs(mu_use)<1e-9: mu_use=0.05
            # path grid
            ns=max(2,int(R0/0.001)+1)
            sp=np.linspace(0,R0,ns)
            rem=R0-sp
            Ea=np.interp(rem, Rinv_E, Rinv_val)
            Sc=interp(Ea,Ee,Scol)
            contrib_base = Sc/abs(mu_use)
            zpos = z0 + mu_use*sp
            idx=np.clip((zpos/dz).astype(int),0,NZ-1)
            # weight per source: n / (N PHI0 V_slab) per unit volume; *dz0=0.2 per unit area
            w = n/Ntot/PHI0/V_slab*0.2
            valid=(zpos>=0)&(zpos<=30)
            if not valid.any(): continue
            np.add.at(D, idx[valid], w*contrib_base[valid])
# Now D(z)/Phi0 in MeV cm^2/g? we expect D(5)~0.0335
print('raw D(5)/Phi0:', D[int(5.0/dz)])
imax=int(np.argmax(D))
print('dmax at', zg[imax], 'Dmax', D[imax])
print('PDD5', 100*D[int(5.0/dz)]/D[imax])
for zz in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0,15.0,20.0]:
    print(f'{zz:5.1f}  {100*D[int(zz/dz)]/D[imax]:7.2f}')