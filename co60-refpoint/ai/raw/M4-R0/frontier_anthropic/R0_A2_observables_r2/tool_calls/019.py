
ana2 = r'''
# ---------- L1 ----------
edges28=np.arange(0,1.4001,0.05)
def rebin(fine, edges_coarse):
    # fine bins width DE from 0; coarse edges multiples of DE
    idx=np.round(edges_coarse/M.DE).astype(int)
    out=np.array([fine[idx[i]:idx[i+1]].sum() for i in range(len(edges_coarse)-1)])
    return out
prim_spec=np.zeros(28)
for E,R in zip(ELINES,R_line):
    prim_spec[int(np.floor(E/0.05))]+=R
scat_spec=rebin(Rs_fine,edges28)
tot_spec=prim_spec+scat_spec
Phi_prim=R_line.sum(); Phi_scat=Rs_fine.sum(); Phi_tot=Phi_prim+Phi_scat
L1_scat_frac=Phi_scat/Phi_tot
Ecen28=(edges28[:-1]+edges28[1:])/2
E_scat_fine=EC
energy_scat=(Rs_fine*EC).sum(); energy_prim=(R_line*ELINES).sum()
L1_scat_energy_frac=energy_scat/(energy_scat+energy_prim)
L1_mean_E=(energy_scat+energy_prim)/Phi_tot
angP=np.zeros(10); angP[9]=1.0
angS=t['fl_cos'][1]/t['fl_cos'][1].sum()
# ---------- L2 ----------
Wi=t['W_int']; We=t['W_etr']
names=['compton','photoelectric','pair','rayleigh']
def fr4(vec):  # vec order coh,inc,pe,pair -> dict order
    v=np.array([vec[1],vec[2],vec[3],vec[0]]); v=v/v.sum()
    return dict(zip(names,[float(x) for x in v]))
L2=fr4(Wi.sum(0)); L2e=fr4(We.sum(0)); L2p=fr4(Wi[0]); L2s=fr4(Wi[1])
L2_share=Wi[1].sum()/Wi.sum()
# ---------- L3 ----------
rec=t['recs']
T_e=rec['T']; c_e=rec['cos']; w_e=rec['w']; f_e=rec['flag']
edges25=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
cedges=np.linspace(-1,1,11)
def hist(x,w,edges):
    h,_=np.histogram(x,bins=edges,weights=w); return h
h3=hist(T_e,w_e,edges25); L3_spec=h3/h3.sum()
m0=f_e==0
h3p=hist(T_e[m0],w_e[m0],edges25); L3_spec_p=h3p/h3p.sum()
L3_mean=(T_e*w_e).sum()/w_e.sum()
L3_mean_p=(T_e[m0]*w_e[m0]).sum()/w_e[m0].sum()
hc=hist(np.clip(c_e,-1,1-1e-12),w_e,cedges); L3_dir=hc/hc.sum()
'''
s=open('/tmp/ana.py').read()+ana2
open('/tmp/ana.py','w').write(s)
exec(s)
print('L1', Phi_prim,Phi_scat, L1_scat_frac, L1_scat_energy_frac, L1_mean_E)
print('L2', L2, L2e, L2p, L2s, L2_share, sep='\n')
print('L3', L3_spec.round(4), L3_mean, L3_mean_p, L3_dir.round(3), sep='\n')
print('scat spec', scat_spec.round(5))
print('angS', angS.round(4))
