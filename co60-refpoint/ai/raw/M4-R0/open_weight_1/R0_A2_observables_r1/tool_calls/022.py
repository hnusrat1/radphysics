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
dz=0.02; zg=np.arange(0,30+dz*0.5,dz); NZ=len(zg)
E_dep=np.zeros(NZ)   # energy deposited per unit area per cell (MeV/cm^2 per Phi0)
for k in range(150):
    z0=0.1+0.2*k; cnt=slab_src[k]
    if cnt.sum()==0: continue
    for b in range(25):
        E0b=emid[b]; R0=interp(E0b,Ee,Rcsda)
        if R0<=0: continue
        for c in range(10):
            mu0=cos_mid[c]; n=cnt[b,c]
            if n<=0: continue
            mu_use=mu0 if abs(mu0)>=0.05 else (0.05 if mu0>=0 else -0.05)
            ns=800; sp=np.linspace(0,R0,ns+1); ds=sp[1]-sp[0]
            spc=0.5*(sp[:-1]+sp[1:]); rem=R0-spc
            Ea=np.interp(rem,Rinv_E,Rinv_val); Sc=interp(Ea,Ee,Scol)
            w = n/Ntot/PHI0/V_slab*0.2   # sources per unit area per Phi0 (cm^-2? we'll see)
            dE = w*Sc*ds                # energy per unit area per source*... MeV cm^-2... per Phi0
            zpos=z0+mu_use*spc
            idx=np.clip((zpos/dz).astype(int),0,NZ-1)
            valid=(zpos>=0)&(zpos<=30)&(dE>0)
            np.add.at(E_dep, idx[valid], dE[valid])
D_dose = E_dep/dz   # MeV/g per Phi0 (since rho=1, per cm^3 = per g; E_dep/dz = MeV/(g cm^2)*... 
# check: E_dep is energy per unit area per Phi0 (MeV cm^-2); /dz (cm) *1/rho -> MeV/g per Phi0?? 
# Dose (MeV/g) = energy per unit mass = (energy per unit area)/(rho dz) = E_dep/dz (rho=1). 
imax=int(np.argmax(D_dose))
print('D(5)/Phi0 (MeV cm^2/g):', D_dose[int(5.0/dz)])
print('D(dmax)/Phi0:', D_dose[imax], 'at', zg[imax])
print('PDD5:', 100*D_dose[int(5.0/dz)]/D_dose[imax])
# compare to Kcol
print('Kcol/Phi0 (MeV cm^2/g):', 5.3690e-12/1.602e-10)
# print profile
for zz in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.25,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0,15.0,20.0]:
    print(f'{zz:5.2f}  {100*D_dose[int(zz/dz)]/D_dose[imax]:6.2f}  D={D_dose[int(zz/dz)]:.5f}')
# total energy conservation check
print('total energy per unit area per Phi0 (E_dep sum * dz):', E_dep.sum()*dz)
print('expected total source energy:', sum([(slab_src[k].sum(1)*emid).sum() for k in range(150)])/Ntot/PHI0/V_slab*0.2)