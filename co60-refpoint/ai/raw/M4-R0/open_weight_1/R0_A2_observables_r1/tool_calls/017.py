import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
flu_p,flu_s,inter_a,inter_p,inter_s,trans,e_all,e_prim,e_cos,e_sum,e_sum_prim,slab_src,Ntot = A
slab_src=slab_src
PHI0=1/10.5**2
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
cos_mid=np.linspace(-0.9,0.9,10)
emid=(el_edges[:-1]+el_edges[1:])/2
s = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=s[:,0]; Scol=s[:,1]; Srad=s[:,2]; Rcsda=s[:,4]
def interp(x,xp,yp): return np.interp(x,xp,yp)
# build range inverse: for a fine grid of path lengths 0..Rmax
Rmax=interp(1.4,Ee,Rcsda)
# dose grid
dz=0.02; zg=np.arange(0,30+dz*0.5,dz); NZ=len(zg)
D=np.zeros(NZ)
V_slab=math.pi*0.2
# source per (slab,e,cos) per unit volume per Phi0: counts/N/PHI0/V_slab  (cm^-1)
# D(z)/Phi0 (MeV cm^2/g): sum over sources: S_src * dz0 * S_c(E(z))/mu
for k in range(150):
    z0=0.1+0.2*k
    cnt=slab_src[k]  # (25,10)
    if cnt.sum()==0: continue
    for b in range(25):
        E0=emid[b]
        R0=interp(E0,Ee,Rcsda)
        if R0<=0: continue
        for c in range(10):
            mu0=cos_mid[c]
            n=cnt[b,c]
            if n<=0: continue
            if abs(mu0)<0.05:  # nearly sideways - treat as straight-ahead deposit at z0? deposit locally over small range
                mu_use=np.sign(mu0)*0.05 if mu0!=0 else 1.0
            else:
                mu_use=mu0
            s_path=np.linspace(0,min(R0,40),2000)
            E_along=np.interp(R0-s_path, Rcsda, Ee, left=0.0, right=0.0)  # E at remaining range; need inverse: E such that R(E)=R0-s
            # actually E_along = energy where remaining range = R0-s: use inverse
            E_along=np.interp(R0-s_path, Rcsda, Ee)  # careful: this is wrong direction; do properly below
            # correct: energy E has range R(E). remaining range rem=R0-s => E = Rinv(rem)
            E_along=np.interp(R0-s_path, np.sort(Rcsda), np.sort(Ee))  # Rinv interpolation: Rcsda increasing with E -> Rinv(rem) increasing as rem increases
            # Hmm R(E) increasing in E => rem = R0-s decreasing with s; E(rem) = inverse. use interp(rem, Rcsda, Ee)
            missing = R0-s_path <= 0
            # deposit: each source electron per unit volume contributes dose S_c(E)/mu per unit mass at depth z=z0+mu*s
            weight_src = n/Ntot/PHI0*0.2  # per unit... times dz0 (0.2 cm) * counts factor? we'll calibrate below
            # contribution to dose array at z indices
            zpos = z0 + mu_use*s_path
            if mu_use>=0:
                valid=(zpos>=0)&(zpos<=30)&(~missing)
            else:
                valid=(zpos>=0)&(zpos<=30)&(~missing)
            if not valid.any(): continue
            Sc = interp(E_along, Ee, Scol)
            contrib = weight_src * Sc / mu_use
            idx=(zpos/dz).astype(int)
            idx=np.clip(idx,0,NZ-1)
            np.add.at(D, idx[valid], contrib[valid])
# calibrate: total dose integrated over depth should equal total collision energy per unit Phi0 per unit area
# total energy from electron sources per unit volume per Phi0 integrated over column volume... Instead calibrate D(5) to Kcol(5)
# Kcol/Phi0 at 5 cm:
print('D(5)/Phi0 (raw MeV cm^2/g):', D[int(5.0/dz)])
print('Kcol/Phi0 (MeV cm^2/g):', 5.368950654924698e-12/1.602176634e-10)
# find dmax
imax=int(np.argmax(D))
print('dmax index depth', zg[imax], 'D(dmax)/Phi0 raw', D[imax])
print('PDD5 (raw D(5)/D(dmax)*100):', 100*D[int(5.0/dz)]/D[imax])
print('PDD10:', 100*D[int(10/dz)]/D[imax])
# profile rough print
for zz in [0.1,0.3,0.5,0.7,1.0,1.5,2.0,3.0,5.0,8.0,10.0]:
    print(zz, 100*D[int(zz/dz)]/D[imax])