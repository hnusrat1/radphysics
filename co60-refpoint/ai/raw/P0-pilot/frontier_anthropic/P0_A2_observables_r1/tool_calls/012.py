
code5 = r'''
import numpy as np, sys, time
sys.path.insert(0,'/tmp/work')
import phys, photon, track, deposit
Om=0.009975072683099257; S_phi0=Om*110.0**2; V=np.pi*1.0**2*0.2
EB=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
CB=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
ZD=np.arange(0.25,20.0001,0.25)
def run_chunk(N,seed):
    res=photon.run_photons(N,seed=seed)
    R=res['r'];D=res['d'];E=res['E'];F=res['f'];L=res['l']
    out={}
    out['N']=N
    out['nproc']=res['nproc']
    # depth-dependent scattered Kcol and fluence
    Kd=np.zeros(len(ZD)); Fd=np.zeros(len(ZD))
    dep=deposit.transfer_coeffs(E)
    fcol=dep['col']; ftr=dep['tr']
    sel_all=np.where(F)[0]
    Rs=R[F];Ds=D[F];Es=E[F];Ls=L[F]
    fcol_s=fcol[F]; ftr_s=ftr[F]
    for k,z in enumerate(ZD):
        lin=track.cyl_track(Rs,Ds,Ls,z)
        Kd[k]=(lin*fcol_s).sum()
        Fd[k]=lin.sum()
    out['Kd']=Kd*S_phi0/V/N     # scattered Kcol per Phi0 (MeV/g per Phi0 before 1.602e-10)
    out['Fd']=Fd*S_phi0/V/N
    # tallies at z=10
    lin=track.cyl_track(Rs,Ds,Ls,10.0)
    m=lin>0
    w=lin[m]
    Em=Es[m]; Dm=Ds[m]
    out['seg']=dict(r=Rs[m],d=Dm,E=Em,l=lin[m])  # in cylinder at z=10
    # process rates (scattered) : sum lin*mu_i
    coh=dep['coh'][F][m]; inc=dep['inc'][F][m]; pe=dep['pe'][F][m]; pr=dep['pr'][F][m]
    out['mu_sc']=np.array([(w*coh).sum(),(w*inc).sum(),(w*pe).sum(),(w*pr).sum()])*S_phi0/V/N
    out['tr_sc']=np.array([(w*dep['tr_c'][F][m]).sum(),(w*dep['tr_pe'][F][m]).sum(),(w*dep['tr_pa'][F][m]).sum()])*S_phi0/V/N
    out['col_sc']=(w*dep['col'][F][m]).sum()*S_phi0/V/N
    out['K_sc']=(w*dep['tr'][F][m]).sum()*S_phi0/V/N
    out['Kcol_sc_check']=(w*dep['col'][F][m]).sum()*S_phi0/V/N
    # fluence spectrum and angular (scattered), per Phi0
    eb=np.clip(np.digitize(Em,EB)-1,-1,27)
    spec=np.zeros(28)
    ok=(eb>=0)&(eb<28)
    np.add.at(spec,eb[ok],w[ok])
    out['spec']=spec*S_phi0/V/N
    out['spec_over']=w[Em>1.4].sum()*S_phi0/V/N
    cb=np.clip(np.digitize(Dm[:,2],CB)-1,0,9)
    ang=np.zeros(10); np.add.at(ang,cb,w)
    out['ang']=ang*S_phi0/V/N
    out['EFl']=(w*Em).sum()*S_phi0/V/N
    out['Fl_sc']=w.sum()*S_phi0/V/N
    # interactions in cylinder by flag all segments at z=10 -- ALSO primary-independent counts
    return out
'''
open('/tmp/work/sim.py','w').write(code5)
import sim
importlib.reload(sim)
t=time.time()
o=sim.run_chunk(200000,11)
print("time",time.time()-t)
print("Fl_sc",o['Fl_sc'],"Kd[:10]",o['Kd'][:10])
print(o['nproc'])
