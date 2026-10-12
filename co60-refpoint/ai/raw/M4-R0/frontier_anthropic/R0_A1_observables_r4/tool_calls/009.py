
mc = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import phys
me=phys.me
Om=phys.Om; F0=phys.F0
R=1.0; HZ=0.1
PH_XY=15.0; PH_Z0=100.0; PH_Z1=130.0
VOL=np.pi*R*R*2*HZ

def unit(a):
    return a/np.linalg.norm(a,axis=1)[:,None]

def rotate(d, c, phi):
    a=np.zeros_like(d); a[:,0]=1.0
    mask=np.abs(d[:,0])>0.9
    a[mask]=np.array([0,1.0,0])
    e1=unit(np.cross(d,a))
    e2=np.cross(d,e1)
    s=np.sqrt(np.clip(1-c*c,0,None))
    out=c[:,None]*d+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
    return unit(out)

def kn_sample(E, rng):
    n=E.size; k=E/me
    eps_out=np.empty(n); c_out=np.empty(n)
    todo=np.arange(n)
    while todo.size>0:
        kk=k[todo]
        c=rng.uniform(-1,1,todo.size)
        eps=1/(1+kk*(1-c))
        f=eps*eps*(eps+1/eps-(1-c*c))
        acc=rng.uniform(0,2.0,todo.size) < f    # f<=2 for all eps in (0,1]
        idx=todo[acc]
        eps_out[idx]=eps[acc]; c_out[idx]=c[acc]
        todo=todo[~acc]
    return eps_out,c_out

def seg_in_cyl(p,d,L,zc):
    dz=d[:,2]; z0=p[:,2]
    lo=zc-HZ; hi=zc+HZ
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(lo-z0)/dz; t2=(hi-z0)/dz
    inside=(z0>=lo)&(z0<=hi)
    tzlo=np.where(dz!=0,np.minimum(t1,t2),np.where(inside,-1e30,1e30))
    tzhi=np.where(dz!=0,np.maximum(t1,t2),np.where(inside,1e30,-1e30))
    dx=d[:,0]; dy=d[:,1]; px=p[:,0]; py=p[:,1]
    a=dx*dx+dy*dy; b=2*(px*dx+py*dy); cc=px*px+py*py-R*R
    disc=b*b-4*a*cc
    with np.errstate(divide='ignore',invalid='ignore'):
        sq=np.sqrt(np.clip(disc,0,None))
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    trlo=np.where(a>1e-14,np.where(disc>0,r1,1e30),np.where(cc<=0,-1e30,1e30))
    trhi=np.where(a>1e-14,np.where(disc>0,r2,-1e30),np.where(cc<=0,1e30,-1e30))
    tlo=np.maximum(np.maximum(0.0,tzlo),trlo)
    thi=np.minimum(np.minimum(L,tzhi),trhi)
    return np.clip(thi-tlo,0,None)

def run(N, seed, mu_fun, cyl_list, Ebins_ph, cos_edges):
    rng=np.random.default_rng(seed)
    nE=len(Ebins_ph)-1
    T={'Lp_E':np.zeros(nE),'Ls_E':np.zeros(nE),'Lp_c':np.zeros(len(cos_edges)-1),'Ls_c':np.zeros(len(cos_edges)-1),
       'Lp':0.0,'Ls':0.0,'LEp':0.0,'LEs':0.0}
    for pr in ['C','pe','pair','coh']:
        T['intp_'+pr]=0.0; T['ints_'+pr]=0.0
        T['trp_'+pr]=0.0; T['trs_'+pr]=0.0
    kerma=np.zeros(len(cyl_list)); flu=np.zeros(len(cyl_list))
    ev=[]
    u=rng.uniform(-0.05,0.05,N); v=rng.uniform(-0.05,0.05,N)
    w=(1+u*u+v*v)**-1.5*phys.A_uv/Om
    p=np.stack([PH_Z0*u,PH_Z0*v,np.full(N,PH_Z0)],1)
    d=unit(np.stack([u,v,np.ones(N)],1))
    E=np.where(rng.uniform(size=N)<0.5,1.1732,1.3325)
    prim=np.ones(N,bool)
    gen=0
    while p.shape[0]>0:
        gen+=1
        n=p.shape[0]
        mp=mu_fun(E)
        mut=mp['tot']
        tex=np.full(n,1e30)
        for ax,(lo,hi) in enumerate([(-PH_XY,PH_XY),(-PH_XY,PH_XY),(PH_Z0,PH_Z1)]):
            da=d[:,ax]
            with np.errstate(divide='ignore',invalid='ignore'):
                tp=np.where(da>0,(hi-p[:,ax])/da,np.where(da<0,(lo-p[:,ax])/da,1e30))
            tex=np.minimum(tex,tp)
        s=-np.log(rng.uniform(size=n))/mut
        esc=s>=tex
        L=np.where(esc,tex,s)
        for ci,zc in enumerate(cyl_list):
            Lin=seg_in_cyl(p,d,L,zc)
            wl=w*Lin
            kerma[ci]+=(wl*(mp['trC']*E+mp['pe']*E+mp['pair']*(E-2*me))).sum()
            flu[ci]+=wl.sum()
            if ci!=0: continue
            sel=np.where(Lin>0)[0]
            if sel.size==0: continue
            pm=prim
            Ebin=np.clip(np.digitize(E,Ebins_ph)-1,0,nE-1)
            cb=np.clip(np.digitize(d[:,2],cos_edges)-1,0,len(cos_edges)-2)
            T['Lp']+=wl[pm].sum(); T['Ls']+=wl[~pm].sum()
            T['LEp']+=(wl*E)[pm].sum(); T['LEs']+=(wl*E)[~pm].sum()
            np.add.at(T['Lp_E'],Ebin[pm],wl[pm]); np.add.at(T['Ls_E'],Ebin[~pm],wl[~pm])
            np.add.at(T['Lp_c'],cb[pm],wl[pm]); np.add.at(T['Ls_c'],cb[~pm],wl[~pm])
            for pr in ['C','pe','pair','coh']:
                T['intp_'+pr]+=(wl*mp[pr])[pm].sum(); T['ints_'+pr]+=(wl*mp[pr])[~pm].sum()
            T['trp_C']+=(wl*mp['trC']*E)[pm].sum(); T['trs_C']+=(wl*mp['trC']*E)[~pm].sum()
            T['trp_pe']+=(wl*mp['pe']*E)[pm].sum(); T['trs_pe']+=(wl*mp['pe']*E)[~pm].sum()
            T['trp_pair']+=(wl*mp['pair']*(E-2*me))[pm].sum(); T['trs_pair']+=(wl*mp['pair']*(E-2*me))[~pm].sum()
            # electron events: Compton (sampled kinematics for each segment in V)
            Es=E[sel]; ds=d[sel]
            eps,cc=kn_sample(Es,rng)
            phi=rng.uniform(0,2*np.pi,sel.size)
            dp=rotate(ds,cc,phi)
            pe_vec=Es[:,None]*ds-(Es*eps)[:,None]*dp
            pn=np.linalg.norm(pe_vec,axis=1)
            ce=pe_vec[:,2]/pn
            Tk=Es*(1-eps)
            wC=wl[sel]*mp['C'][sel]
            ev.append(np.stack([Tk,ce,wC,pm[sel].astype(float),np.zeros(sel.size),np.zeros(sel.size)],1))
            # photoelectric
            ip=sel[mp['pe'][sel]>0]
            if ip.size>0:
                ev.append(np.stack([E[ip],rng.uniform(-1,1,ip.size),wl[ip]*mp['pe'][ip],pm[ip].astype(float),np.ones(ip.size),np.zeros(ip.size)],1))
            ipp=sel[mp['pair'][sel]>0]
            if ipp.size>0:
                f=rng.uniform(size=ipp.size); Tt=E[ipp]-2*me
                wPP=wl[ipp]*mp['pair'][ipp]
                for TT in [f*Tt,(1-f)*Tt]:
                    ev.append(np.stack([TT,np.full(ipp.size,0.9),wPP,pm[ipp].astype(float),np.full(ipp.size,2.0),np.zeros(ipp.size)],1))
        # ---- interactions of non-escaping photons
        idx=np.where(~esc)[0]
        newp=[];newd=[];newE=[];newpr=[];neww=[]
        if idx.size>0:
            pint=p[idx]+s[idx][:,None]*d[idx]
            Ei=E[idx]; di=d[idx]; pri=prim[idx]; wi=w[idx]
            mpi={k:v[idx] for k,v in mp.items()}
            r=rng.uniform(size=idx.size)*mpi['tot']
            isC=r<mpi['C']
            isCoh=(~isC)&(r<mpi['C']+mpi['coh'])
            isPE=(~isC)&(~isCoh)&(r<mpi['C']+mpi['coh']+mpi['pe'])
            isPair=(~isC)&(~isCoh)&(~isPE)
            ic=np.where(isC)[0]
            if ic.size>0:
                eps,cc=kn_sample(Ei[ic],rng)
                dn=rotate(di[ic],cc,rng.uniform(0,2*np.pi,ic.size))
                En=Ei[ic]*eps
                ok=En>=0.001
                newp.append(pint[ic][ok]); newd.append(dn[ok]); newE.append(En[ok]); newpr.append(np.zeros(ok.sum(),bool)); neww.append(wi[ic][ok])
            ir=np.where(isCoh)[0]
            if ir.size>0:
                cr=np.empty(ir.size); todo=np.arange(ir.size)
                while todo.size>0:
                    c=rng.uniform(-1,1,todo.size)
                    acc=rng.uniform(0,1,todo.size)< (1+c*c)/2
                    cr[todo[acc]]=c[acc]; todo=todo[~acc]
                dn=rotate(di[ir],cr,rng.uniform(0,2*np.pi,ir.size))
                newp.append(pint[ir]); newd.append(dn); newE.append(Ei[ir]); newpr.append(np.zeros(ir.size,bool)); neww.append(wi[ir])
            ip2=np.where(isPair & (Ei>=2*me))[0]
            if ip2.size>0:
                cz=rng.uniform(-1,1,ip2.size); ph=rng.uniform(0,2*np.pi,ip2.size)
                s_=np.sqrt(1-cz*cz)
                dd=np.stack([s_*np.cos(ph),s_*np.sin(ph),cz],1)
                for sg in [1,-1]:
                    newp.append(pint[ip2]); newd.append(sg*dd); newE.append(np.full(ip2.size,me)); newpr.append(np.zeros(ip2.size,bool)); neww.append(wi[ip2])
        if len(newp)>0:
            p=np.concatenate(newp); d=np.concatenate(newd); E=np.concatenate(newE); prim=np.concatenate(newpr); w=np.concatenate(neww)
        else:
            break
        if gen>200: break
    ev=np.concatenate(ev) if len(ev)>0 else np.zeros((0,6))
    return T,kerma,flu,ev
A_uv=phys.A_uv
'''
open('/tmp/work/mc.py','w').write(mc)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, mc as MC, numpy as np, phys, time
cos_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
Eph=np.arange(0,1.4001,0.05)
t0=time.time()
T,kerma,flu,ev=MC.run(20000,1,phys.mu_parts,[105.0,100.5],Eph,cos_edges)
print(time.time()-t0, T['Lp']/MC.VOL*MC.F0, T['Ls']/MC.VOL*MC.F0, flu)
