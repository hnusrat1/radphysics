import json
# sample already defined from previous cell - but we need to combine all in one cell
# Let me just reconstruct the events by re-running the simulation once more to get the final answer

# Actually, the sample from the previous cell is no longer available. Let me re-run everything in this cell.

import numpy as np, math, time, json
from collections import Counter

rng = np.random.default_rng(12345)
me = 0.51099895; re = 2.8179403262e-13; Ne = 3.343e23

def kn_f(mu,E):
    a=E/me; r=1.0/(1.0+a*(1.0-mu))
    return r*r*(1.0/r+r-(1.0-mu*mu))

Egrid=np.logspace(np.log10(0.001),np.log10(10.0),300)
sig_grid=np.empty(300)
for j in range(300):
    E=Egrid[j]; mus=np.linspace(-1,1,3001)
    sig_grid[j]=np.pi*re**2*np.trapezoid(kn_f(mus,E),mus)

E_ph=np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
mu_ph=np.array([4.90,0.506,0.162,0.0712,0.0375,0.0223,0.00988,0.00547,0.00240,0.00128,0.000526,0.000151,5e-5,3e-5,1e-5,4.7e-6])
E_pa=np.array([1.022,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0])
mu_pa=np.array([0.0,3.1e-5,1.32e-4,4.1e-4,8.1e-4,1.39e-3,2.69e-3,4.32e-3,6.19e-3,1.05e-2,1.55e-2])
E_co=np.array([0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50,0.80,1.0,1.5,2.0])
mu_co=np.array([0.30,0.060,0.022,0.0098,0.0046,0.0034,0.0017,0.0011,0.0006,0.0003,0.00015,0.00012,8e-5,4.5e-5])

def mu_comp(E): return Ne*np.interp(np.log(np.clip(E,1e-3,10)),np.log(Egrid),sig_grid)
def mu_photo(E): return np.exp(np.interp(np.log(np.clip(E,E_ph[0],E_ph[-1])),np.log(E_ph),np.log(mu_ph)))
def mu_pair(E):
    res=np.zeros_like(E); mask=E>1.022
    if mask.any():
        res[mask]=np.exp(np.interp(np.log(np.clip(E[mask],E_pa[1],E_pa[-1])),np.log(E_pa[1:]),np.log(mu_pa[1:])))
    return res
def mu_coh(E): return np.exp(np.interp(np.log(np.clip(E,E_co[0],E_co[-1])),np.log(E_co),np.log(mu_co)))

def sample_compton(E):
    n=len(E); mu=np.empty(n); alive=np.ones(n,bool); a=E/me
    while alive.any():
        idx=np.where(alive)[0]; mu_try=rng.uniform(-1,1,len(idx)); f=kn_f(mu_try,E[idx])
        acc=rng.uniform(0,2,len(idx))<f; mu[idx[acc]]=mu_try[acc]; alive[idx[acc]]=False
    Ep=E/(1.0+a*(1.0-mu)); te=E-Ep; num=E-Ep*mu; den=np.sqrt(E*E+Ep*Ep-2.0*E*Ep*mu)
    cos_el=np.where(den>1e-15,num/den,1.0)
    return mu,Ep,te,cos_el

def rotate_dir(ux,uy,uz,mu_s,phi):
    d=np.column_stack([ux,uy,uz]); d=d/np.linalg.norm(d,axis=1,keepdims=True)
    ref=np.tile(np.array([0.,0.,1.]),(len(d),1)); nz=np.abs(d[:,2])>0.9; ref[nz]=[1.,0.,0.]
    e1=np.cross(d,ref); e1=e1/(np.linalg.norm(e1,axis=1,keepdims=True)+1e-30)
    e2=np.cross(d,e1); st=np.sqrt(np.clip(1.0-mu_s*mu_s,0,1))
    nd=(st*np.cos(phi))[:,np.newaxis]*e1+(st*np.sin(phi))[:,np.newaxis]*e2+mu_s[:,np.newaxis]*d
    nd=nd/np.linalg.norm(nd,axis=1,keepdims=True); return nd[:,0],nd[:,1],nd[:,2]

def inside_cube(x,y,z): return (np.abs(x)<15)&(np.abs(y)<15)&(z>100)&(z<130)
def in_scoring(x,y,z): return (x*x+y*y<1.0)&(z>109.9)&(z<110.1)

def run_batch(n_prim,ev_out):
    x=rng.uniform(-5,5,n_prim); y=rng.uniform(-5,5,n_prim); z=np.full(n_prim,100.0)
    d=np.sqrt(x*x+y*y+z*z); ux=x/d; uy=y/d; uz=z/d; E=np.full(n_prim,2.0); alive=np.ones(n_prim,bool)
    ex=[];ey=[];ez=[];eux=[];euy=[];euz=[];eE=[]
    for _ in range(80):
        n=np.sum(alive); 
        if n==0: break
        idx=np.where(alive)[0]; Ea=E[idx]; uxa=ux[idx]; uya=uy[idx]; uza=uz[idx]; xa=x[idx]; ya=y[idx]; za=z[idx]
        mc=mu_comp(Ea); mp=mu_photo(Ea); mpp=mu_pair(Ea); mco=mu_coh(Ea); mall=mc+mp+mpp+mco
        s=-np.log(rng.uniform(1e-10,1,n))/mall; xa+=s*uxa; ya+=s*uya; za+=s*uza
        inside=inside_cube(xa,ya,za); in_idx=idx[inside]; n_in=len(in_idx)
        if n_in>0:
            Ei=E[in_idx]; uxi=uxa[inside]; uyi=uya[inside]; uzi=uza[inside]; xi=xa[inside]; yi=ya[inside]; zi=za[inside]
            mci=mc[inside]; mpi=mp[inside]; mppi=mpp[inside]; mcoi=mco[inside]; malli=mall[inside]
            scoring=in_scoring(xi,yi,zi)
            u=rng.uniform(0,malli,n_in); cum_c=mci; cum_ph=mci+mpi; cum_pa=mci+mpi+mppi
            proc=np.full(n_in,3); proc[u<=cum_c]=0; proc[(u>cum_c)&(u<=cum_ph)]=1; proc[(u>cum_ph)&(u<=cum_pa)]=2
            # compton
            mask=proc==0
            if mask.any():
                idx_c=in_idx[mask]; Ec=Ei[mask]; mus,Ep,te,cel=sample_compton(Ec); scm=scoring[mask]
                if scm.any():
                    for i in np.where(scm)[0]:
                        ev_out.append({'process':'compton','e_in_MeV':float(Ec[i]),'e_out_MeV':float(Ep[i]),'cos_scatter':float(mus[i]),'t_e_MeV':float(te[i]),'cos_electron':float(cel[i]),'t_positron_MeV':None})
                phi=rng.uniform(0,2*np.pi,len(Ec)); unx,uny,unz=rotate_dir(uxi[mask],uyi[mask],uzi[mask],mus,phi)
                E[idx_c]=Ep; ux[idx_c]=unx; uy[idx_c]=uny; uz[idx_c]=unz; x[idx_c]=xi[mask]; y[idx_c]=yi[mask]; z[idx_c]=zi[mask]
            # photo
            mask=proc==1
            if mask.any():
                idx_pe=in_idx[mask]; Epe=Ei[mask]; scm=scoring[mask]
                if scm.any():
                    for i in np.where(scm)[0]:
                        ev_out.append({'process':'photoelectric','e_in_MeV':float(Epe[i]),'e_out_MeV':None,'cos_scatter':None,'t_e_MeV':float(Epe[i]),'cos_electron':float(rng.uniform(-1,1)),'t_positron_MeV':None})
                alive[idx_pe]=False
            # pair
            mask=proc==2
            if mask.any():
                idx_pa=in_idx[mask]; Epa=Ei[mask]; scm=scoring[mask]
                tavail=np.maximum(Epa-1.022,0.0); share=rng.uniform(0,1,len(Epa)); t_pos=share*tavail; t_e=tavail-t_pos
                Etot=t_e+me; th0=me/np.clip(Etot,1e-10,None); th_e=th0*(-np.log(rng.uniform(1e-10,1,len(Epa)))); c_el=np.cos(th_e)
                if scm.any():
                    for i in np.where(scm)[0]:
                        ev_out.append({'process':'pair','e_in_MeV':float(Epa[i]),'e_out_MeV':None,'cos_scatter':None,'t_e_MeV':float(t_e[i]),'cos_electron':float(c_el[i]),'t_positron_MeV':float(t_pos[i])})
                for k in range(len(Epa)):
                    for _ in range(2):
                        u1=rng.uniform(-1,1); u2=rng.uniform(0,2*np.pi)
                        nx=math.sqrt(1-u1*u1)*math.cos(u2); ny=math.sqrt(1-u1*u1)*math.sin(u2)
                        ex.append(xi[mask][k]); ey.append(yi[mask][k]); ez.append(zi[mask][k]); eux.append(nx); euy.append(ny); euz.append(u1); eE.append(0.511)
                alive[idx_pa]=False
            # rayleigh
            mask=proc==3
            if mask.any():
                idx_r=in_idx[mask]; Er=Ei[mask]; scm=scoring[mask]
                th0=np.clip(0.0007/Er,1e-6,0.5); th_r=th0*(-np.log(rng.uniform(1e-10,1,len(Er)))); mu_s_r=np.cos(th_r)
                if scm.any():
                    for i in np.where(scm)[0]:
                        ev_out.append({'process':'rayleigh','e_in_MeV':float(Er[i]),'e_out_MeV':float(Er[i]),'cos_scatter':float(mu_s_r[i]),'t_e_MeV':0.0,'cos_electron':None,'t_positron_MeV':None})
                phi=rng.uniform(0,2*np.pi,len(Er)); unx,uny,unz=rotate_dir(uxi[mask],uyi[mask],uzi[mask],mu_s_r,phi)
                E[idx_r]=Er; ux[idx_r]=unx; uy[idx_r]=uny; uz[idx_r]=unz; x[idx_r]=xi[mask]; y[idx_r]=yi[mask]; z[idx_r]=zi[mask]
        alive[in_idx]=True; out_idx=idx[~inside]; alive[out_idx]=False
        if len(ex)>0:
            x=np.concatenate([x,np.array(ex)]); y=np.concatenate([y,np.array(ey)]); z=np.concatenate([z,np.array(ez)])
            ux=np.concatenate([ux,np.array(eux)]); uy=np.concatenate([uy,np.array(euy)]); uz=np.concatenate([uz,np.array(euz)])
            E=np.concatenate([E,np.array(eE)]); alive=np.concatenate([alive,np.ones(len(ex),bool)])
            ex=[];ey=[];ez=[];eux=[];euy=[];euz=[];eE=[]
        if np.sum(~alive)>len(alive)//2:
            k=np.where(alive)[0]; x=x[k]; y=y[k]; z=z[k]; ux=ux[k]; uy=uy[k]; uz=uz[k]; E=E[k]; alive=alive[k]
    return

events=[]
t0=time.time()
run_batch(1500000, events)
t1=time.time()
print(f"Done: {len(events)} events in {t1-t0:.1f}s")
print(Counter(e['process'] for e in events))

# Subsample 200 uniformly
rng2=np.random.default_rng(777)
idx=np.random.choice(len(events),200,replace=False)
final_events=[events[i] for i in idx]
print("Final sample:", Counter(e['process'] for e in final_events))

# Build JSON output
output = {
    "method": "Analog Monte Carlo transport of 2 MeV photons (SSD=100 cm, 10x10 cm field) through 30 cm water cube. Cross sections: Klein-Nishina (Compton), NIST-based (photoelectric, pair, Rayleigh). Photons tracked with exponential step sampling; interactions in scoring cylinder (r=1 cm, thickness 0.2 cm, at 10 cm depth) recorded. 1.5 million primaries yielded 684 interactions; 200 uniformly subsampled.",
    "events": final_events
}

print(json.dumps(output, indent=2)[:2000])
# Write to file for later use
with open('/tmp/result.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nJSON written to /tmp/result.json")