
import numpy as np, time, os

me=0.51099895; re=2.8179403262e-13; NA=6.02214076e23; n_e_g=NA*10/18.01528
def kn_total(E):
    eta=np.asarray(E,float)/me; a=1+2*eta
    return 2*np.pi*re**2*((1+eta)/eta**2*(2*(1+eta)/a-np.log(a)/eta)+np.log(a)/(2*eta)-(1+3*eta)/a**2)
Egrid=np.array([0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0,3.0])
muR=np.array([0.028,0.012,0.0085,0.006,0.0052,0.0045,0.0033,0.0028,0.0022,0.0019,0.0017,0.0015,0.0012,0.001,0.0009,0.00085,0.0008,0.00075,0.00055,0.0004])
tau=np.array([0.6,0.15,0.045,0.0093,0.0026,0.0019,0.0007,0.00042,0.00016,0.000085,0.000052,0.000036,0.000021,0.000013,0.00001,0.000009,0.000008,0.000007,0.000004,0.0000015])
kap=np.zeros_like(Egrid); sel=Egrid>1.022; kap[sel]=np.interp(np.clip(Egrid[sel],1.1732,3.0),[1.1732,1.3325,1.5,2.0,3.0],[0.00012,0.00075,0.0011,0.0026,0.0055])
def interp1(x,xp,fp): return np.interp(np.clip(x,xp[0],xp[-1]),xp,fp)
def mu_comp(E): return kn_total(np.clip(E,0.01,3.0))*n_e_g
def mu_ray(E): return interp1(E,Egrid,muR)
def mu_photo(E): return interp1(E,Egrid,tau)
def mu_pair(E):
    E=np.asarray(E,float); return np.where(E>2*me, interp1(E,Egrid,kap),0.0)
def mu_totX(E): return mu_comp(E)+mu_ray(E)+mu_photo(E)+mu_pair(E)

# energy absorption coefficient water
Eeg=np.array([0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.1732,1.25,1.3325,1.5,2.0])
menv=np.array([0.147,0.088,0.063,0.0410,0.0270,0.0255,0.0276,0.0299,0.0319,0.0326,0.0330,0.0327,0.0320,0.0310,0.0301,0.0297,0.0292,0.0283,0.0261])
def mu_en(E): return interp1(E,Eeg,menv)

eta_grid=np.linspace(0.02,3.0,400); xi_grid=np.linspace(0.001,0.999,512); mu_grid=np.linspace(-1,1,4001)
def kn_dsdmu_vec(eta,mu):
    r=1.0/(1.0+eta*(1.0-mu)); return np.pi*re**2*r**2*(r+1.0/r-(1.0-mu*mu))
Q=np.zeros((len(eta_grid),len(xi_grid)))
for i,eta in enumerate(eta_grid):
    d=kn_dsdmu_vec(eta,mu_grid); c=np.cumsum(d); c=c/c[-1]; Q[i,:]=np.interp(xi_grid,c,mu_grid)
def sample_kn_cos(E,xi):
    eta=np.clip(E/me,eta_grid[0],eta_grid[-1])
    ie=(eta-eta_grid[0])/(eta_grid[-1]-eta_grid[0])*(len(eta_grid)-1)
    i0=np.clip(np.floor(ie).astype(int),0,len(eta_grid)-2); f=ie-i0
    ix=xi*(len(xi_grid)-1); j0=np.clip(np.floor(ix).astype(int),0,len(xi_grid)-2); fx=ix-j0
    v0=Q[i0,j0]*(1-fx)+Q[i0,j0+1]*fx; v1=Q[i0+1,j0]*(1-fx)+Q[i0+1,j0+1]*fx
    return v0*(1-f)+v1*f
def rotate(u,cosT,phi):
    ux,uy,uz=u[...,0],u[...,1],u[...,2]
    e1x=uy; e1y=-ux; e1z=np.zeros_like(ux); n=np.sqrt(e1x*e1x+e1y*e1y)
    e1x=np.where(n>1e-30,e1x/n,1.0); e1y=np.where(n>1e-30,e1y/n,0.0)
    e2x=uy*e1z-uz*e1y; e2y=uz*e1x-ux*e1z; e2z=ux*e1y-uy*e1x
    st=np.sqrt(np.clip(1-cosT*cosT,0,1)); cphi=np.cos(phi); sphi=np.sin(phi)
    nx=e1x*(st*cphi)+e2x*(st*sphi)+ux*cosT; ny=e1y*(st*cphi)+e2y*(st*sphi)+uy*cosT; nz=e1z*(st*cphi)+e2z*(st*sphi)+uz*cosT
    return np.stack([nx,ny,nz],axis=-1)
def score_cylinder(x,y,z,ux,uy,uz,s,zlo,zhi,rad=1.0):
    uzp=np.where(uz!=0,uz,np.nan)
    tz1=(zlo-z)/uzp; tz2=(zhi-z)/uzp
    tza=np.minimum(tz1,tz2); tzb=np.maximum(tz1,tz2)
    tza=np.clip(tza,0.0,s); tzb=np.clip(tzb,0.0,s)
    uz0=np.abs(uz)<1e-9
    tza=np.where(uz0,0.0,tza); tzb=np.where(uz0,s,tzb)
    tza=np.where(uz0&((z<zlo)|(z>zhi)),0.0,tza); tzb=np.where(uz0&((z<zlo)|(z>zhi)),0.0,tzb)
    a=ux*ux+uy*uy; b=2*(x*ux+y*uy); c=x*x+y*y-rad*rad
    disc=b*b-4*a*c; a0=np.abs(a)<1e-9
    sq=np.sqrt(np.clip(disc,0,None)); den=2*a+1e-30
    t1=(-b-sq)/den; t2=(-b+sq)/den; neg=disc<0
    tlo=np.where(neg,0.0,np.minimum(t1,t2)); thi=np.where(neg,np.where(c<0,s,0.0),np.maximum(t1,t2))
    start_in=c<0
    tlo=np.where(start_in&~neg&~a0,0.0,tlo); thi=np.where(start_in&~neg&~a0,np.maximum(t1,t2),thi)
    tlo=np.where(a0,0.0,tlo); thi=np.where(a0,np.where(c<0,s,0.0),thi)
    tlo=np.maximum(tlo,tza); thi=np.minimum(thi,tzb)
    return np.maximum(thi-tlo,0.0)

ph_bins=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ct_bins=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

def run_chunk(N, seed, Hp, Hcnt, stats, depth=5.0):
    rng=np.random.default_rng(seed)
    x0=rng.uniform(-5,5,N); y0=rng.uniform(-5,5,N)
    d=np.sqrt(x0*x0+y0*y0+1e4)
    ux=x0/d; uy=y0/d; uz=100.0/d
    E=np.where(rng.random(N)<0.5,1.1732,1.3325)
    x=x0.copy(); y=y0.copy(); z=np.zeros(N)
    scat=np.zeros(N,bool); alive=np.ones(N,bool); pending=[]
    while True:
        if not alive.any():
            if len(pending)==0: break
            arr=np.concatenate(pending,axis=0); pending=[]
            x=np.concatenate([x,arr[:,0]]); y=np.concatenate([y,arr[:,1]]); z=np.concatenate([z,arr[:,2]])
            ux=np.concatenate([ux,arr[:,3]]); uy=np.concatenate([uy,arr[:,4]]); uz=np.concatenate([uz,arr[:,5]])
            E=np.concatenate([E,arr[:,6]]); scat=np.concatenate([scat,np.ones(len(arr),bool)])
            alive=np.concatenate([alive,np.ones(len(arr),bool)])
        Ncur=len(alive)
        mu=mu_totX(E)
        s=-np.log(rng.random(Ncur))/mu; s=np.where(alive,s,0.0)
        uzp=np.where(uz!=0,uz,np.nan); uxp=np.where(ux!=0,ux,np.nan); uyp=np.where(uy!=0,uy,np.nan)
        tex=np.full(Ncur,np.inf)
        tz=(30-z)/uzp; ok=uz>0; tex=np.where(ok,np.minimum(tex,np.where(tz>0,tz,np.inf)),tex)
        tz=(0-z)/uzp; ok=uz<0; tex=np.where(ok,np.minimum(tex,np.where(tz>0,tz,np.inf)),tex)
        tx=(15-x)/uxp; ok=ux>0; tex=np.where(ok,np.minimum(tex,np.where(tx>0,tx,np.inf)),tex)
        tx=(-15-x)/uxp; ok=ux<0; tex=np.where(ok,np.minimum(tex,np.where(tx>0,tx,np.inf)),tex)
        ty=(15-y)/uyp; ok=uy>0; tex=np.where(ok,np.minimum(tex,np.where(ty>0,ty,np.inf)),tex)
        ty=(-15-y)/uyp; ok=uy<0; tex=np.where(ok,np.minimum(tex,np.where(ty>0,ty,np.inf)),tex)
        s_eff=np.minimum(s,tex); s_eff=np.where(alive,s_eff,0.0)
        tl=score_cylinder(x,y,z,ux,uy,uz,s_eff,depth-0.1,depth+0.1)*alive
        eb=np.clip(np.searchsorted(ph_bins,E,side='right')-1,0,27)
        cb=np.clip(np.searchsorted(ct_bins,uz,side='right')-1,0,9)
        cat=scat.astype(int)
        idx=eb*20+cb*2+cat
        Hp += np.bincount(idx, weights=tl, minlength=560)
        Hcnt[0]+=tl[~scat].sum(); Hcnt[1]+=tl[scat].sum()
        interact=(s<tex)&alive
        x=x+ux*s*alive; y=y+uy*s*alive; z=z+uz*s*alive
        dead=alive&(s>=tex)
        if interact.any():
            ind=np.where(interact)[0]
            xi=x[ind]; yi=y[ind]; zi=z[ind]; ui=np.stack([ux[ind],uy[ind],uz[ind]],axis=-1); Ei=E[ind]
            mui=mu_totX(Ei)
            pc=mu_comp(Ei)/mui; pr=pc+mu_ray(Ei)/mui; pp=pr+mu_photo(Ei)/mui
            r=rng.random(len(ind))
            mC=r<pc; mR=(r>=pc)&(r<pr); mPh=(r>=pr)&(r<pp); mPa=(r>=pp)
            if mC.any():
                cosT=sample_kn_cos(Ei[mC],rng.random(mC.sum())); phi=2*np.pi*rng.random(mC.sum())
                un=rotate(ui[mC],cosT,phi); ii=ind[mC]
                ux[ii]=un[:,0]; uy[ii]=un[:,1]; uz[ii]=un[:,2]
                E[ii]=Ei[mC]/(1.0+(Ei[mC]/me)*(1.0-cosT)); scat[ii]=True
            if mR.any():
                cosT=1.0-2e-5*rng.random(mR.sum()); phi=2*np.pi*rng.random(mR.sum())
                un=rotate(ui[mR],cosT,phi); ii=ind[mR]
                ux[ii]=un[:,0]; uy[ii]=un[:,1]; uz[ii]=un[:,2]; scat[ii]=True
            if mPh.any(): dead[ind[mPh]]=True
            if mPa.any():
                npa=mPa.sum(); dead[ind[mPa]]=True
                th=rng.random(npa)*np.pi; ph=rng.random(npa)*2*np.pi
                d1=np.stack([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)],axis=-1); d2=-d1
                for dd in (d1,d2):
                    pending.append(np.column_stack([xi[mPa],yi[mPa],zi[mPa],dd[:,0],dd[:,1],dd[:,2],np.full(npa,0.511)]))
            lowE=E[ind]<0.01; dead[ind[lowE]]=True
            if dead.any():
                E[dead]=0.0; ux[dead]=0; uy[dead]=0; uz[dead]=0
        alive=(~dead)&alive
    stats['events']+=0

t0=time.time()
Hp=np.zeros(560); Hcnt=np.zeros(2); stats={'events':0}
Ntot=12_000_000; chunk=2_000_000
for i in range(int(np.ceil(Ntot/chunk))):
    Nc=min(chunk,Ntot-i*chunk)
    run_chunk(Nc, 5000+i, Hp, Hcnt, stats, depth=5.0)
    print('chunk',i,'done','time',round(time.time()-t0,1),'Hcnt',Hcnt,flush=True)
print('TOTAL time',round(time.time()-t0,1))
Vcyl=np.pi*0.2
fp=Hcnt[0]/Vcyl/Ntot*110.25; fs=Hcnt[1]/Vcyl/Ntot*110.25
print('primary fluence/Phi0',fp,'scattered',fs,'total',fp+fs,'scatfrac',fs/(fp+fs))
sp_p=Hp.reshape(28,10,2)[:,:,0].sum(axis=1)/Vcyl/Ntot*110.25
sp_s=Hp.reshape(28,10,2)[:,:,1].sum(axis=1)/Vcyl/Ntot*110.25
print('Kcol/Phi0 (Gy cm2):', (sp_p+sp_s)* (np.arange(28)*0.05+0.025) ) # placeholder
Ec = 0.5*(ph_bins[1:]+ph_bins[:-1])
Kcol_prim=((sp_p*Ec*mu_en(Ec)).sum()*1.602176634e-10)
Kcol_scat=((sp_s*Ec*mu_en(Ec)).sum()*1.602176634e-10)
print('Kcol_prim Gy cm2',Kcol_prim,'Kcol_scat',Kcol_scat,'Kcol_total',Kcol_prim+Kcol_scat)
print('mean photon energy primary', (sp_p*Ec).sum()/fp, 'scattered', (sp_s*Ec).sum()/fs)
ang_p=Hp.reshape(28,10,2)[:,:,0].sum(axis=0)/max(Hcnt[0],1e-30)
ang_s=Hp.reshape(28,10,2)[:,:,1].sum(axis=0)/max(Hcnt[1],1e-30)
print('ang prim',np.round(ang_p,4))
print('ang scat',np.round(ang_s,4))
np.savez('mc5.npz',Hp=Hp,Hcnt=Hcnt,Ntot=Ntot,sp_p=sp_p,sp_s=sp_s,ang_p=ang_p,ang_s=ang_s,Kcol_prim=Kcol_prim,Kcol_scat=Kcol_scat)
