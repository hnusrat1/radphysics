
mc = r'''
import numpy as np, time
from scipy.integrate import quad
import phys
from phys import mu_coh, mu_inc, mu_pe, mu_pair, mu_tot_fn, mec2
R=1.0; ZLO=4.9; ZHI=5.1
DZS=0.1; NSLAB=100  # slabs 0..10 cm
NEB=280; DE=0.005   # fine energy bins 0..1.4 MeV
# KN energy transfer fraction table
def fC_exact(E):
    k=E/mec2
    def eps(mu): return 1/(1+k*(1-mu))
    def dsig(mu):
        e=eps(mu); return e*e*(e+1/e-(1-mu*mu))
    num,_=quad(lambda mu: dsig(mu)*(1-eps(mu)), -1,1)
    den,_=quad(lambda mu: dsig(mu), -1,1)
    return num/den
_Eg=np.logspace(-3,np.log10(1.5),200)
_fg=np.array([fC_exact(e) for e in _Eg])
def fC(E):
    return np.interp(np.log(E), np.log(_Eg), _fg)
def mu_tr_coef(E):
    # energy-transfer coefficient (per cm) from Compton, photoelectric, pair (kinetic)
    pr=np.where(E>1.022, (E-1.022)/E, 0.0)
    return mu_inc(E)*fC(E) + mu_pe(E) + mu_pair(E)*pr
def clip_cyl(p0,d,L,zlo,zhi,R):
    z0=p0[:,2]; dz=d[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        s1=(zlo-z0)/dz; s2=(zhi-z0)/dz
    sa=np.where(np.abs(dz)>1e-12, np.minimum(s1,s2), np.where((z0>=zlo)&(z0<=zhi),-1e9,1e9))
    sb=np.where(np.abs(dz)>1e-12, np.maximum(s1,s2), np.where((z0>=zlo)&(z0<=zhi),1e9,-1e9))
    x0=p0[:,0]; y0=p0[:,1]; dx=d[:,0]; dy=d[:,1]
    a=dx*dx+dy*dy; b=2*(x0*dx+y0*dy); c=x0*x0+y0*y0-R*R
    disc=b*b-4*a*c
    ok=disc>=0
    sq=np.sqrt(np.maximum(disc,0))
    with np.errstate(divide='ignore',invalid='ignore'):
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    ra=np.where(a>1e-14, np.minimum(r1,r2), np.where(c<=0,-1e9,1e9))
    rb=np.where(a>1e-14, np.maximum(r1,r2), np.where(c<=0,1e9,-1e9))
    ra=np.where(ok|(a<=1e-14),ra,1e9); rb=np.where(ok|(a<=1e-14),rb,-1e9)
    lo=np.maximum.reduce([np.zeros_like(L), sa, ra])
    hi=np.minimum.reduce([L, sb, rb])
    TL=np.maximum(hi-lo,0.0)
    return TL, lo, hi
def pyr_omega():
    g,w=np.polynomial.legendre.leggauss(60)
    u=0.05*g; wu=0.05*w
    U,V=np.meshgrid(u,u); W=np.outer(wu,wu)
    return np.sum(W*(1+U**2+V**2)**-1.5)
OMEGA=pyr_omega()
def run(N, seed, tallies):
    rng=np.random.default_rng(seed)
    # sample pyramid directions uniform in solid angle (rejection)
    uu=[];vv=[]
    need=N
    while need>0:
        m=int(need*1.05)+10
        u=rng.uniform(-0.05,0.05,m); v=rng.uniform(-0.05,0.05,m)
        acc=rng.random(m)< (1+u*u+v*v)**-1.5
        uu.append(u[acc]); vv.append(v[acc]); need-=acc.sum()
    u=np.concatenate(uu)[:N]; v=np.concatenate(vv)[:N]
    n=u.size
    nrm=np.sqrt(1+u*u+v*v)
    d=np.stack([u/nrm,v/nrm,1/nrm],1)
    pos=np.stack([100*u,100*v,np.zeros(n)],1)
    E=np.where(rng.random(n)<0.5,1.1732,1.3325)
    flag=np.zeros(n,dtype=np.int8)   # 0 primary, 1 scattered
    # active photons
    t0=time.time()
    gen=0
    while pos.shape[0]>0:
        gen+=1
        m=pos.shape[0]
        mu=mu_tot_fn(E)
        s_int=-np.log(rng.random(m))/mu
        # exit distance
        with np.errstate(divide='ignore',invalid='ignore'):
            tx=np.where(d[:,0]>0,(15-pos[:,0])/d[:,0],np.where(d[:,0]<0,(-15-pos[:,0])/d[:,0],np.inf))
            ty=np.where(d[:,1]>0,(15-pos[:,1])/d[:,1],np.where(d[:,1]<0,(-15-pos[:,1])/d[:,1],np.inf))
            tz=np.where(d[:,2]>0,(30-pos[:,2])/d[:,2],np.where(d[:,2]<0,(0-pos[:,2])/d[:,2],np.inf))
        t_exit=np.minimum(np.minimum(tx,ty),tz)
        L=np.minimum(s_int,t_exit)
        # tallies in cylinder
        TL,lo,hi=clip_cyl(pos,d,L,ZLO,ZHI,R)
        sel=TL>0
        if sel.any():
            Es=E[sel]; fl=flag[sel]; dz=d[sel,2]; TLs=TL[sel]
            tallies['fl_E']+=np.bincount(fl*NEB+np.minimum((Es/DE).astype(int),NEB-1), weights=TLs, minlength=2*NEB).reshape(2,NEB)
            cb=np.clip(np.digitize(dz,np.linspace(-1,1,11))-1,0,9)
            tallies['fl_cos']+=np.bincount(fl*10+cb, weights=TLs, minlength=20).reshape(2,10)
            mus=[mu_coh(Es),mu_inc(Es),mu_pe(Es),mu_pair(Es)]
            fr=[np.zeros_like(Es), fC(Es)*Es, Es, np.where(Es>1.022,Es-1.022,0.0)]
            for p in range(4):
                tallies['W_int'][:,p]+=np.bincount(fl,weights=TLs*mus[p],minlength=2)
                tallies['W_etr'][:,p]+=np.bincount(fl,weights=TLs*mus[p]*fr[p]/np.maximum(Es,1e-12),minlength=2)
            # electron records (segment-based expected interactions)
            mtr=mu_tr_coef(Es)
            # Compton
            wC=TLs*mus[1]; wP=TLs*mus[2]; wPa=TLs*mus[3]
            k=Es/mec2
            epsC=phys.kn_eps_sample(Es,rng)
            Tc=Es*(1-epsC)
            cth=1-(1/epsC-1)/k
            phi=2*np.pi*rng.random(Es.size)
            # build outgoing direction
            din=d[sel]
            # orthonormal basis perp to din
            a=np.where(np.abs(din[:,0])<0.9,np.array([1,0,0]),np.array([0,1,0]))
            e1=np.cross(din,a); e1/=np.linalg.norm(e1,axis=1)[:,None]
            e2=np.cross(din,e1)
            sth=np.sqrt(np.maximum(0,1-cth**2))
            dout=(din*cth[:,None]+e1*(sth*np.cos(phi))[:,None]+e2*(sth*np.sin(phi))[:,None])
            Eout=Es*epsC
            pe_vec=Es[:,None]*din-Eout[:,None]*dout
            pn=np.linalg.norm(pe_vec,axis=1)+1e-30
            cos_e=pe_vec[:,2]/pn
            rec=tallies['recs']
            rec['T'].append(Tc); rec['cos'].append(cos_e); rec['w'].append(wC); rec['flag'].append(fl.astype(np.int8)); rec['proc'].append(np.zeros(Es.size,np.int8))
            rec['T'].append(Es.copy()); rec['cos'].append(din[:,2].copy()); rec['w'].append(wP); rec['flag'].append(fl.astype(np.int8)); rec['proc'].append(np.ones(Es.size,np.int8))
            Tp=np.where(Es>1.022,Es-1.022,0.0); T1=Tp*rng.random(Es.size)
            for part in [T1,Tp-T1]:
                rec['T'].append(part); rec['cos'].append(din[:,2].copy()); rec['w'].append(wPa); rec['flag'].append(fl.astype(np.int8)); rec['proc'].append(np.full(Es.size,2,np.int8))
            tallies['slab_TL']+=0  # placeholder
            # slab tallies (depth dose)
            slab_add(tallies, pos[sel], d[sel], lo[sel], hi[sel], Es, fl, mtr, TLs)
        # move
        newpos=pos+L[:,None]*d
        inter = (s_int < t_exit)
        # process photons that interact
        idx=np.where(inter)[0]
        keep_pos=[];keep_d=[];keep_E=[];keep_f=[]
        if idx.size:
            Ei=E[idx]; mui=mu[idx]; pi=newpos[idx]; di=d[idx]; fi=flag[idx]
            r=rng.random(idx.size)*mui
            pc=mu_coh(Ei); pinc=mu_inc(Ei); ppe=mu_pe(Ei); ppa=mu_pair(Ei)
            c0=pc; c1=c0+pinc; c2=c1+ppe
            isC=(r>=c0)&(r<c1); isR=r<c0; isPE=(r>=c1)&(r<c2); isPA=r>=c2
            # Rayleigh
            if isR.any():
                ii=np.where(isR)[0]
                mu_s=np.empty(ii.size); todo=np.arange(ii.size)
                while todo.size:
                    c=2*rng.random(todo.size)-1
                    acc=rng.random(todo.size)*2<(1+c*c)
                    mu_s[todo[acc]]=c[acc]; todo=todo[~acc]
                dn=_scatter_dir(di[ii],mu_s,rng)
                keep_pos.append(pi[ii]); keep_d.append(dn); keep_E.append(Ei[ii]); keep_f.append(np.ones(ii.size,np.int8))
            if isC.any():
                ii=np.where(isC)[0]
                eps=phys.kn_eps_sample(Ei[ii],rng)
                cth=1-(1/eps-1)/(Ei[ii]/mec2)
                dn=_scatter_dir(di[ii],cth,rng)
                keep_pos.append(pi[ii]); keep_d.append(dn); keep_E.append(Ei[ii]*eps); keep_f.append(np.ones(ii.size,np.int8))
            if isPA.any():
                ii=np.where(isPA)[0]
                ok=Ei[ii]>1.022
                ii=ii[ok]
                if ii.size:
                    cz=2*rng.random(ii.size)-1; ph=2*np.pi*rng.random(ii.size); st=np.sqrt(1-cz*cz)
                    dn=np.stack([st*np.cos(ph),st*np.sin(ph),cz],1)
                    keep_pos.append(pi[ii]); keep_d.append(dn); keep_E.append(np.full(ii.size,0.511)); keep_f.append(np.ones(ii.size,np.int8))
                    keep_pos.append(pi[ii]); keep_d.append(-dn); keep_E.append(np.full(ii.size,0.511)); keep_f.append(np.ones(ii.size,np.int8))
            # photoelectric: absorbed (no new photon)
        # photons transported without interaction (not interacting, not exited)
        cont=np.where(~inter)[0]
        if cont.size:
            pass
        # assemble next generation: continuing photons = interacted-and-scattered ones
        newp=[];newd=[];newE=[];newf=[]
        if idx.size and keep_pos:
            newp=np.concatenate(keep_pos); newd=np.concatenate(keep_d); newE=np.concatenate(keep_E); newf=np.concatenate(keep_f)
            # kill low energy
            live=newE>0.001
            newp=newp[live]; newd=newd[live]; newE=newE[live]; newf=newf[live]
        pos=newp if len(newp) else np.zeros((0,3)); d=newd if len(newp) else np.zeros((0,3))
        E=newE if len(newp) else np.zeros(0); flag=newf if len(newp) else np.zeros(0,np.int8)
        if len(newp)==0: break
    return
def _scatter_dir(din,cth,rng):
    a=np.where((np.abs(din[:,0])<0.9)[:,None],np.array([1.0,0,0]),np.array([0,1.0,0]))
    e1=np.cross(din,a); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(din,e1)
    ph=2*np.pi*rng.random(din.shape[0])
    sth=np.sqrt(np.maximum(0,1-cth**2))
    dn=din*cth[:,None]+e1*(sth*np.cos(ph))[:,None]+e2*(sth*np.sin(ph))[:,None]
    return dn/np.linalg.norm(dn,axis=1)[:,None]
def slab_add(tallies, p, d, lo, hi, Es, fl, mtr, TLs):
    # distribute segment pieces [lo,hi] (s-units) into depth slabs 0..10 cm (z from 0)
    z0=p[:,2]; dz=d[:,2]
    ok=(hi>lo)&(np.abs(dz)>1e-12)
    if not ok.any(): return
    z0=z0[ok]; dz=dz[ok]; lo=lo[ok]; hi=hi[ok]; Es=Es[ok]; fl=fl[ok]; mtr=mtr[ok]
    za=z0+lo*dz; zb=z0+hi*dz
    zmin=np.minimum(za,zb); zmax=np.maximum(za,zb)
    kmin=np.floor(zmin/DZS).astype(int); kmax=np.floor(zmax/DZS).astype(int)
    kmin=np.clip(kmin,0,NSLAB-1); kmax=np.clip(kmax,0,NSLAB-1)
    span=int((kmax-kmin).max())+1 if kmax.size else 0
    for j in range(span):
        k=kmin+j
        valid=k<=kmax
        zk=k*DZS; zk1=zk+DZS
        s1=(zk-z0)/dz; s2=(zk1-z0)/dz
        sa=np.minimum(s1,s2); sb=np.maximum(s1,s2)
        ov=np.maximum(0,np.minimum(hi,sb)-np.maximum(lo,sa))
        ov=np.where(valid,ov,0.0)
        np.add.at(tallies['slab_TL'], (fl.astype(int), k), ov)
        np.add.at(tallies['slab_K'], (fl.astype(int), k), ov*mtr*Es)
'''
open('/tmp/mc.py','w').write(mc)
import importlib, sys
sys.path.insert(0,'/tmp')
import mc as M
import time
def new_tallies():
    return dict(fl_E=np.zeros((2,M.NEB)), fl_cos=np.zeros((2,10)), W_int=np.zeros((2,4)), W_etr=np.zeros((2,4)),
                slab_TL=np.zeros((2,M.NSLAB)), slab_K=np.zeros((2,M.NSLAB)),
                recs=dict(T=[],cos=[],w=[],flag=[],proc=[]))
t=new_tallies()
t0=time.time()
M.run(20000,1,t)
print('time',time.time()-t0)
print(t['fl_E'].sum(axis=1), t['W_int'])
