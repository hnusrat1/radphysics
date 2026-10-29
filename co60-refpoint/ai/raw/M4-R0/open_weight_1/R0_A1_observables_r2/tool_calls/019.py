import sys, os
sys.path.insert(0, os.getcwd())
import mc_photon as mcp
import numpy as np, time
# instrumented copy
def run_instr(N, seed=1):
    rng = np.random.default_rng(seed)
    nph=mcp.nph; nct=mcp.nct; nel=mcp.nel
    flu_pri_ph=np.zeros(nph); flu_sct_ph=np.zeros(nph)
    n_cross=0; Lsct_total=0; Lsct_events=[]; n_sct_events=0
    x=rng.uniform(-5,5,N); y=rng.uniform(-5,5,N); z=np.zeros(N)
    dx=x; dy=y; dz=100.0; norm=np.sqrt(dx*dx+dy*dy+dz*dz)
    pos=np.stack([x,y,z],axis=1); dire=np.stack([dx/norm,dy/norm,dz/norm],axis=1)
    E=np.where(rng.random(N)<0.5,1.1732,1.3325)
    alive=np.ones(N,bool); primary=np.ones(N,bool)
    Ecut=0.002; it=0
    while alive.any():
        it+=1
        act_idx=np.where(alive)[0]; n_act=len(act_idx)
        Em=np.maximum(E[alive],Ecut)
        mu=mcp.mu_tot(Em)
        s=-np.log(np.maximum(rng.random(n_act),1e-30))/mu
        p1=pos[alive]+dire[alive]*s[:,None]
        L=mcp.seg_in_cyl_vec(pos[alive],p1)
        m=L>0
        if m.any():
            n_cross+=int(m.sum())
            pri=primary[alive][m]
            Lsct_events.append(L[m][~pri] if (~pri).any() else np.array([]))
            n_sct_events+=int((~pri).sum())
            Lsct_total+=L[m][~pri].sum() if (~pri).any() else 0
            ph_idx=np.clip(np.searchsorted(mcp.ph_edges,Em[m],side='right')-1,0,nph-1)
            np.add.at(flu_sct_ph,ph_idx[~pri],L[m][~pri])
            np.add.at(flu_pri_ph,ph_idx[pri],L[m][pri])
        outside=(p1[:,2]<0)|(p1[:,2]>30)|(np.abs(p1[:,0])>15)|(np.abs(p1[:,1])>15)
        pos[alive]=p1
        comp_p=mcp.mu_comp(Em)/mu; coh_p=mcp.mu_coh(Em)/mu; pho_p=mcp.mu_pho(Em)/mu
        rr=rng.random(n_act)
        comp_a=rr<comp_p; coh_a=(~comp_a)&(rr<comp_p+coh_p); pho_a=(~comp_a)&(~coh_a)&(rr<comp_p+coh_p+pho_p); pri_a=(~comp_a)&(~coh_a)&(~pho_a)
        if comp_a.any():
            Ecomp=Em[comp_a]
            costh=mcp.sample_compton_cos(Ecomp, rng.random(int(comp_a.sum())))
            Ep=Ecomp/(1+(Ecomp/mcp.mec2)*(1-costh))
            u_old=dire[alive][comp_a]
            cross=np.cross(u_old,np.array([0.,0.,1.])); nc=np.linalg.norm(cross,axis=1)
            fall=nc<1e-8; e1=np.zeros_like(cross); e1[~fall]=cross[~fall]/nc[~fall,None]; e1[fall]=np.array([1.,0.,0.])
            e2=np.cross(u_old,e1); phi=rng.uniform(0,2*np.pi,int(comp_a.sum()))
            sinth=np.sqrt(np.maximum(1-costh*costh,0))
            u_new=u_old*costh[:,None]+(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)*sinth[:,None]
            set_idx=act_idx[comp_a]; E[set_idx]=Ep; dire[set_idx]=u_new; primary[set_idx]=False
        if coh_a.any():
            Eco=Em[coh_a]; costh=mcp.sample_coherent_cos(Eco, rng.random(int(coh_a.sum())))
            u_old=dire[alive][coh_a]
            cross=np.cross(u_old,np.array([0.,0.,1.])); nc=np.linalg.norm(cross,axis=1)
            fall=nc<1e-8; e1=np.zeros_like(cross); e1[~fall]=cross[~fall]/nc[~fall,None]; e1[fall]=np.array([1.,0.,0.])
            e2=np.cross(u_old,e1); phi=rng.uniform(0,2*np.pi,int(coh_a.sum()))
            sinth=np.sqrt(np.maximum(1-costh*costh,0))
            u_new=u_old*costh[:,None]+(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)*sinth[:,None]
            dire[act_idx[coh_a]]=u_new; primary[act_idx[coh_a]]=False
        if pho_a.any(): alive[act_idx[pho_a]]=False
        if pri_a.any(): alive[act_idx[pri_a]]=False
        alive[act_idx[outside]]=False
        alive[alive&(E<Ecut)]=False
    return dict(n_cross=n_cross, n_sct_events=n_sct_events, Lsct_total=Lsct_total, flu_pri_ph=flu_pri_ph, flu_sct_ph=flu_sct_ph, Lsct_events=Lsct_events)
r = run_instr(10000, seed=5)
print('n_cross', r['n_cross'], 'n_sct_events', r['n_sct_events'], 'Lsct_total', r['Lsct_total'])
allL = np.concatenate([a for a in r['Lsct_events'] if len(a)])
print('max L', allL.max() if len(allL) else 0, 'mean', allL.mean() if len(allL) else 0)
print('sct flu per phi0', r['flu_sct_ph'].sum()/mcp.Vcyl/10000*mcp.Phi0_area)
print('sct spectrum per phi0', np.round(r['flu_sct_ph']/mcp.Vcyl/10000*mcp.Phi0_area,4))
