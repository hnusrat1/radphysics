
mc = r'''
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import phys
me=phys.me
Om=phys.Om; F0=phys.F0
ZC=105.0; R=1.0; HZ=0.1
PH_XY=15.0; PH_Z0=100.0; PH_Z1=130.0

def rotate(d, c, phi):
    # d (n,3) unit; c cos, phi azimuth
    n=d.shape[0]
    a=np.zeros_like(d); a[:,0]=1.0
    mask=np.abs(d[:,0])>0.9
    a[mask]=np.array([0,1.0,0])
    e1=np.cross(d,a); e1/=np.linalg.norm(e1,axis=1)[:,None]
    e2=np.cross(d,e1)
    s=np.sqrt(np.clip(1-c*c,0,None))
    out=c[:,None]*d+s[:,None]*(np.cos(phi)[:,None]*e1+np.sin(phi)[:,None]*e2)
    out/=np.linalg.norm(out,axis=1)[:,None]
    return out

def kn_sample(E, rng):
    # returns eps=E'/E and cos of scattering angle (lab), rejection on cos
    E=np.asarray(E,float); n=E.size
    k=E/me
    eps_out=np.empty(n); c_out=np.empty(n)
    todo=np.arange(n)
    # envelope: f(c)=eps^2(eps+1/eps-sin^2) <= 2*eps^2 * ... use max 4? compute bound numerically per E: use 2.0*(1+2k)^? use generous bound
    while todo.size>0:
        kk=k[todo]
        c=rng.uniform(-1,1,todo.size)
        eps=1/(1+kk*(1-c))
        f=eps*eps*(eps+1/eps-(1-c*c))
        fmax=np.maximum(2.0,4.0*np.ones_like(f))  # f<=~ 2.0 at most for eps in [0,1]
        acc=rng.uniform(0,1,todo.size)*fmax < f
        idx=todo[acc]
        eps_out[idx]=eps[acc]; c_out[idx]=c[acc]
        todo=todo[~acc]
    return eps_out,c_out

def seg_in_cyl(p,d,L,zc):
    dz=d[:,2]; z0=p[:,2]
    lo_z=zc-HZ; hi_z=zc+HZ
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(lo_z-z0)/dz; t2=(hi_z-z0)/dz
    tzlo=np.where(dz!=0,np.minimum(t1,t2),np.where((z0>=lo_z)&(z0<=hi_z),-1e30,1e30))
    tzhi=np.where(dz!=0,np.maximum(t1,t2),np.where((z0>=lo_z)&(z0<=hi_z),1e30,-1e30))
    dx=d[:,0]; dy=d[:,1]; px=p[:,0]; py=p[:,1]
    a=dx*dx+dy*dy; b=2*(px*dx+py*dy); cc=px*px+py*py-R*R
    disc=b*b-4*a*cc
    with np.errstate(divide='ignore',invalid='ignore'):
        sq=np.sqrt(np.clip(disc,0,None))
        r1=(-b-sq)/(2*a); r2=(-b+sq)/(2*a)
    trlo=np.where(a>0,np.where(disc>0,r1,1e30),np.where(cc<=0,-1e30,1e30))
    trhi=np.where(a>0,np.where(disc>0,r2,-1e30),np.where(cc<=0,1e30,-1e30))
    tlo=np.maximum.reduce([np.zeros_like(L),tzlo,trlo])
    thi=np.minimum.reduce([L,tzhi,trhi])
    return np.clip(thi-tlo,0,None)

def run(N, seed, mu_fun, Ebins_ph, Ebins_e, cos_edges, Ebins_fine, cyl_list=[ZC]):
    rng=np.random.default_rng(seed)
    out={}
    # tallies: arrays for main cylinder (index 0 in cyl_list)
    nE=len(Ebins_ph)-1
    tal={k:np.zeros(nE) for k in ['Lp_E','Ls_E']}
    tal.update({k:np.zeros(len(cos_edges)-1) for k in ['Lp_c','Ls_c']})
    tal.update({'Lp':0.0,'Ls':0.0,'LEp':0.0,'LEs':0.0})
    proc=['C','pe','pair','coh']
    for pr in proc:
        tal['intp_'+pr]=0.0; tal['ints_'+pr]=0.0
        tal['trp_'+pr]=0.0; tal['trs_'+pr]=0.0
    # per-depth energy-transfer (kerma) and fluence for each cylinder
    kerma=np.zeros(len(cyl_list)); flu=np.zeros(len(cyl_list))
    ev=[]  # electron events: (T, cos_e, weight, prim_flag, proc_index)
    # start
    u=rng.uniform(-0.05,0.05,N); v=rng.uniform(-0.05,0.05,N)
    w=(1+u*u+v*v)**-1.5*A_UV/Om
    p=np.stack([PH_Z0*u,PH_Z0*v,np.full(N,PH_Z0)],1)
    d=np.stack([u,v,np.ones(N)],1); d/=np.linalg.norm(d,axis=1)[:,None]
    E=np.where(rng.uniform(size=N)<0.5,1.1732,1.3325)
    prim=np.ones(N,bool)
    # arrays of active photons
    while p.shape[0]>0:
        n=p.shape[0]
        mp=mu_fun(E)
        mut=mp['tot']
        # distance to exit
        tex=np.full(n,1e30)
        for ax,(lo,hi) in enumerate([(-PH_XY,PH_XY),(-PH_XY,PH_XY),(PH_Z0,PH_Z1)]):
            da=d[:,ax]
            with np.errstate(divide='ignore',invalid='ignore'):
                tp=np.where(da>0,(hi-p[:,ax])/da,np.where(da<0,(lo-p[:,ax])/da,1e30))
            tex=np.minimum(tex,tp)
        s=-np.log(rng.uniform(size=n))/mut
        esc=s>=tex
        L=np.where(esc,tex,s)
        # score segment
        for ci,zc in enumerate(cyl_list):
            Lin=seg_in_cyl(p,d,L,zc)
            if ci==0:
                sel=Lin>0
                if np.any(sel):
                    wl=w*Lin
                    Ebin=np.clip(np.digitize(E,Ebins_ph)-1,0,nE-1)
                    cb=np.clip(np.digitize(d[:,2],cos_edges)-1,0,len(cos_edges)-2)
                    pm=prim
                    tal['Lp']+=wl[pm].sum(); tal['Ls']+=wl[~pm].sum()
                    tal['LEp']+=(wl*E)[pm].sum(); tal['LEs']+=(wl*E)[~pm].sum()
                    np.add.at(tal['Lp_E'],Ebin[pm],wl[pm]); np.add.at(tal['Ls_E'],Ebin[~pm],wl[~pm])
                    np.add.at(tal['Lp_c'],cb[pm],wl[pm]); np.add.at(tal['Ls_c'],cb[~pm],wl[~pm])
                    for pr in proc:
                        mpr=mp[pr] if pr!='coh' else mp['coh']
                        tal['intp_'+pr]+=(wl*mpr)[pm].sum(); tal['ints_'+pr]+=(wl*mpr)[~pm].sum()
                    trC=mp['trC']; pe=mp['pe']; pair=mp['pair']
                    trtot=trC*E  # energy-transfer rate (MeV/cm) for compton
                    tal['trp_C']+=(wl*trC*E)[pm].sum(); tal['trs_C']+=(wl*trC*E)[~pm].sum()
                    tal['trp_pe']+=(wl*pe*E)[pm].sum(); tal['trs_pe']+=(wl*pe*E)[~pm].sum()
                    tal['trp_pair']+=(wl*pair*(E-2*me))[pm].sum(); tal['trs_pair']+=(wl*pair*(E-2*me))[~pm].sum()
                    kerma[0]+= (wl*(trC*E+pe*E+pair*(E-2*me))).sum()
                    flu[0]+= wl.sum()
                    # electron events from this segment (sampled per expected interaction)
                    idx=np.where(sel)[0]
                    # Compton electron sample
                    eps,cc=kn_sample(E[idx],rng)
                    Ts=E[idx]*(1-eps)
                    # scattered direction and electron momentum
                    phi=rng.uniform(0,2*np.pi,idx.size)
                    dp=rotate(d[idx],cc,phi)
                    pe_vec=E[idx][:,None]*d[idx]-(E[idx]*eps)[:,None]*dp
                    pn=np.linalg.norm(pe_vec,axis=1)
                    ce=pe_vec[:,2]/pn
                    wC=wl[idx]*mp['C'][idx]/np.maximum(mp['C'][idx],1e-300)*mp['C'][idx]/mp['C'][idx]  # weight = wl*muC
                    wC=wl[idx]*mp['C'][idx]
                    ev.append(np.stack([Ts,ce,wC,pm[idx].astype(float),np.zeros(idx.size)],1))
                    # photoelectric electron
                    ip=np.where(sel & (mp['pe']>0))[0]
                    if ip.size>0:
                        wP=wl[ip]*mp['pe'][ip]
                        ce_p=rng.uniform(-1,1,ip.size)
                        ev.append(np.stack([E[ip],ce_p,wP,pm[ip].astype(float),np.ones(ip.size)],1))
                    ipp=np.where(sel & (mp['pair']>0))[0]
                    if ipp.size>0:
                        wPP=wl[ipp]*mp['pair'][ipp]
                        T=E[ipp]-2*me; f=rng.uniform(size=ipp.size)
                        ev.append(np.stack([f*T,np.ones(ipp.size)*0.9,wPP,pm[ipp].astype(float),2*np.ones(ipp.size)],1))
                        ev.append(np.stack([(1-f)*T,np.ones(ipp.size)*0.9,wPP,pm[ipp].astype(float),2*np.ones(ipp.size)],1))
            else:
                kerma[ci]+=(w*Lin*(mp['trC']*E+mp['pe']*E+mp['pair']*(E-2*me))).sum()
                flu[ci]+=(w*Lin).sum()
        # interactions
        keep=~esc
        if not np.any(keep):
            break
        idx=np.where(keep)[0]
        p_int=p[idx]+s[idx][:,None]*d[idx]
        E_i=E[idx]; d_i=d[idx]; pr_i=prim[idx]
        mp_i={k:v[idx] for k,v in mp.items()}
        r=rng.uniform(size=idx.size)*mp_i['tot']
        isC=r<mp_i['C']
        r2=r-mp_i['C']
        isCoh=(~isC)&(r2<mp_i['coh'])
        r3=r2-mp_i['coh']
        isPE=(~isC)&(~isCoh)&(r3<mp_i['pe'])
        isPair=(~isC)&(~isCoh)&(~isPE)
        new_p=[];new_d=[];new_E=[];new_pr=[]
        alive=np.zeros(idx.size,bool)
        # Compton
        ic=np.where(isC)[0]
        if ic.size>0:
            eps,cc=kn_sample(E_i[ic],rng)
            phi=rng.uniform(0,2*np.pi,ic.size)
            dn=rotate(d_i[ic],cc,phi)
            En=E_i[ic]*eps
            alive_c=En>=0.001
            new_p.append(p_int[ic][alive_c]); new_d.append(dn[alive_c]); new_E.append(En[alive_c]); new_pr.append(np.zeros(alive_c.sum(),bool))
        # Rayleigh
        ir=np.where(isCoh)[0]
        if ir.size>0:
            cr=np.empty(ir.size); todo=np.arange(ir.size)
            while todo.size>0:
                c=rng.uniform(-1,1,todo.size)
                acc=rng.uniform(size=todo.size)*1.0 < (1+c*c)/2
                cr[todo[acc]]=c[acc]; todo=todo[~acc]
            phi=rng.uniform(0,2*np.pi,ir.size)
            dn=rotate(d_i[ir],cr,phi)
            new_p.append(p_int[ir]); new_d.append(dn); new_E.append(E_i[ir]); new_pr.append(np.zeros(ir.size,bool))
        # photoelectric: absorbed; electron tally handled by segment-level sampling
        # pair: annihilation photons
        ip2=np.where(isPair & (E_i>=2*me))[0]
        if ip2.size>0:
            cz=rng.uniform(-1,1,ip2.size); ph=rng.uniform(0,2*np.pi,ip2.size)
            s_=np.sqrt(1-cz*cz)
            dd=np.stack([s_*np.cos(ph),s_*np.sin(ph),cz],1)
            for sgn in [1,-1]:
                new_p.append(p_int[ip2]); new_d.append(sgn*dd); new_E.append(np.full(ip2.size,me)); new_pr.append(np.zeros(ip2.size,bool))
        if len(new_p)>0:
            p=np.concatenate(new_p); d=np.concatenate(new_d); E=np.concatenate(new_E); prim=np.concatenate(new_pr)
            w=np.concatenate([w[idx][ic][alive_c] if False else np.zeros(0)]) if False else None
        else:
            p=np.zeros((0,3)); d=np.zeros((0,3)); E=np.zeros(0); prim=np.zeros(0,bool)
        # weights: carry per-photon weight
        if w is None or True:
            pass
        break
    return out
A_UV=phys.A_uv
'''
open('/tmp/work/mc.py','w').write(mc)
print('written')
