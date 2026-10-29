import numpy as np, pickle
with open('/tmp/photon_results.pkl','rb') as f:
    res = pickle.load(f)
d = np.load('/tmp/photon_sources.npz')
src_pos=d['pos']; src_dir=d['dir']; src_T=d['T']; src_par=d['par']; src_kind=d['kind']
print('src', src_pos.shape, 'T range', src_T.min(), src_T.max(), 'mean', src_T.mean())
print('par frac primary', (src_par==1).mean())
print('kind frac positron', (src_kind==1).mean())
# T distribution
for lo,hi in [(0,0.1),(0.1,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0),(1.0,1.2)]:
    print(f'T {lo}-{hi}: {((src_T>=lo)&(src_T<hi)).mean():.4f}')
# positions in region: z range, r range
r=np.sqrt(src_pos[:,0]**2+src_pos[:,1]**2)
print('z min/max', src_pos[:,2].min(), src_pos[:,2].max(), 'r max', r.max())
N=6000000
V= np.pi*1.0**2*0.2
Omega=4*np.arcsin(np.sin(np.arctan(5/100))**2)
Phi0=1/(Omega*105**2)
print('Phi0 per emitted', Phi0)
fl_prim=res['fl_prim']; fl_scat=res['fl_scat']
L1_prim = fl_prim.sum()/(N*V)/Phi0
L1_tot = (fl_prim.sum()+fl_scat.sum())/(N*V)/Phi0
print('L1_primary_fluence_per_phi0', L1_prim, 'analytic 0.7295')
print('L1_total_fluence_per_phi0', L1_tot)
print('L1 scattered frac flu', fl_scat.sum()/(fl_prim.sum()+fl_scat.sum()))
Edges=np.linspace(0,1.4,29); Ec=(Edges[:-1]+Edges[1:])/2
Efl_prim=(Ec*fl_prim).sum(); Efl_scat=(Ec*fl_scat).sum()
print('L1 scattered frac energy flu', Efl_scat/(Efl_prim+Efl_scat))
print('L1 mean photon E', (Ec*(fl_prim+fl_scat)).sum()/(fl_prim+fl_scat).sum())
print('L3 n', len(res['L3_Te']), 'mean Te', res['L3_Te'].mean())
p=res['L3_parent']==1
print('mean Te primary parent', res['L3_Te'][p].mean())