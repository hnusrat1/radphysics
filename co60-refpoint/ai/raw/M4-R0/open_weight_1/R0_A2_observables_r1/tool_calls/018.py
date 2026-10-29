import numpy as np, math
A = np.load('/tmp/mc_tallies_5e6.npy', allow_pickle=True)
slab_src=A[11]; Ntot=A[12]
PHI0=1/10.5**2
s = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=s[:,0]; Scol=s[:,1]; Rcsda=s[:,4]
def interp(x,xp,yp): return np.interp(x,xp,yp)
print('slab_src[25].sum()=',slab_src[25].sum(), 'per bin average', slab_src[25].mean())
print('slab_src[0].sum()=',slab_src[0].sum(), 'slab_src[150-1].sum()=',slab_src[-1].sum())
print('total slab_src',slab_src.sum())
# V_slab
V_slab=math.pi*0.2
el_edges=np.array([0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
emid=(el_edges[:-1]+el_edges[1:])/2
cos_mid=np.linspace(-0.9,0.9,10)
# verify D(5) analytic: sources per unit volume per Phi0 = counts/N/PHI0/V_slab
# Let me just compute the local electron source strength at slab 25 (z~5)
cnt=slab_src[25]
src_vol_per_bin = cnt/Ntot/PHI0/V_slab   # per bin (cm^-1, i.e. per unit Phi0 per volume)
print('src per volume per Phi0 per bin at slab25 (cm^-1): ', src_vol_per_bin.sum())
print('= like q̃ integral per cm? compare q̃ integral 0.0854 over 25 bins...')
# q̃ per bin (integrated) from MC: sources per cm^3 per Phi0 per bin = cnt/Ntot/PHI0/V_slab
# q̃ integral was 0.0854 cm^-1 = electrons per cm^3 per Phi0? check with cn

# simple check of D formula normalization: local equilibrium dose ~ Kcol
# Kcol/Phi0 = 0.03351 MeV cm^2/g. In equilibrium D ≈ Σ_sources E0 per unit mass: 
# sources per unit volume per Phi0 (sum over bins, per cm) * mean initial energy = energy per cm^3 per Phi0 -> *1/rho = MeV/g... need /cm^2 from Phi0? 
print('energy source per cm^3 per Phi0 at slab25:', (src_vol_per_bin*emid).sum(), 'MeV/cm^3 per Phi0')
print('Kcol at 5cm in MeV/cm^3 per Phi0 (should equal this in equilibrium):', 0.03351)
# ratio
print('ratio source/Kcol:', (src_vol_per_bin*emid).sum()/0.03351)