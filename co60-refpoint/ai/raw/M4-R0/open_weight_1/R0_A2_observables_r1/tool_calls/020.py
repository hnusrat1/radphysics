import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
slab_src=A[11]; Ntot=int(A[12])
PHI0=1/10.5**2
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
emid=(el_edges[:-1]+el_edges[1:])/2
V_slab=math.pi*0.2
cnt=slab_src[25]   # (25,10)
Svol = cnt/Ntot/PHI0/V_slab
print('Svol total at slab25:', Svol.sum())
print('energy-weighted Svol*E0 sum:', (Svol.sum(1)*emid).sum())
print('compare Kcol/Phi0 (MeV cm^2/g):', 5.369e-12/1.602e-10)
# total over all slabs
Eall = (slab_src.sum(2).sum(1)*emid).sum()   # energy-weighted counts all slabs
print('total energy-weighted counts all slabs:', Eall)
print('total source energy per unit Phi0 per cm^3 * dz0 summed:', Eall/Ntot/PHI0/V_slab*0.2)
# Also expected: primary Kcol integrated along axis = Kcol(0)* integral exp(-mu z) etc. just info
# check factor: 0.0335 vs Svol*E0
print('ratio', (Svol.sum(1)*emid).sum()/ (5.369e-12/1.602e-10))