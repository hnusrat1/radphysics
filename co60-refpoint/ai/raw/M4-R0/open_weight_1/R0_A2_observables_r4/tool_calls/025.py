import numpy as np, time
from scipy.interpolate import interp1d
from scipy.integrate import quad

# ================= electron physics =================
el=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
Ee=el[:,0]; Sc_=el[:,1]; Sr_=el[:,2]; St_=el[:,3]; R_=el[:,4]; Y_=el[:,5]
def Scol(E): return np.interp(E,Ee,Sc_)
def Srad(E): return np.interp(E,Ee,Sr_)
def Yrad(E): return np.interp(E,Ee,Y_)
mec2=0.5109989461
r0=2.8179403262e-13
Ne_per_g=3.34283e23
Delta=0.010
Egrid=np.logspace(np.log10(0.0105),np.log10(1.5),600)
def moler_pref(E):
    gam=1+E/mec2; b2=1-1/(gam*gam); return 2*np.pi*r0*r0*mec2/b2
def dsigma(E,T):
    gam=1+E/mec2; A=moler_pref(E)
    return A*(1/T**2+1/(E-T)**2-(2*gam-1)/(gam*gam)/(T*(E-T)))
S_rest_tab=np.zeros_like(Egrid); Ndel_tab=np.zeros_like(Egrid); Sdel_tab=np.zeros_like(Egrid)
for j,E in enumerate(Egrid):
    if E<=2*Delta:
        S_rest_tab[j]=Scol(E); Ndel_tab[j]=0; Sdel_tab[j]=0
    else:
        sd=quad(lambda T: T*dsigma(E,T),Delta,E/2)[0]
        nd=quad(lambda T: dsigma(E,T),Delta,E/2)[0]
        Sdel_tab[j]=Ne_per_g*sd; Ndel_tab[j]=Ne_per_g*nd
        S_rest_tab[j]=max(Scol(E)-Ne_per_g*sd,0.05)
def Srest(E): return np.interp(E,Egrid,S_rest_tab)
def Scol_f(E): return np.interp(E,Ee,Sc_)
def Srad_f(E): return np.interp(E,Ee,Sr_)
def Ndel(E): return np.interp(E,Egrid,Ndel_tab)

def sample_delta_energy(E,n,rng):
    Tmax=E/2; out=np.empty(n); got=np.zeros(n,bool)
    invD=1/Delta; norm=1/Delta-1/Tmax
    A=moler_pref(E); gam=1+E/mec2
    while not got.all():
        u=rng.uniform(0,1,n)
        T=1.0/(1/Delta - u*norm)
        w=1+(T*T)/(E-T)**2 - (2*gam-1)/(gam*gam)*T/(E-T)
        acc=rng.uniform(0,1,n)*1.02 < w
        new=(~got)&acc; out[new]=T[new]; got[new]=True
    return out

def highland_sigma(E, ds):
    X0=36.08
    pc=np.sqrt(E*(E+2*mec2)); Etot=E+mec2; beta=pc/Etot
    betacp=pc*beta
    if betacp<1e-6: return 0.0
    x=ds/X0
    return min((13.6/betacp)*np.sqrt(x)*(1+0.038*np.log(max(x,1e-6))),1.5)

eb_edges=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
ct_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
nEB=len(eb_edges)-1; nCT=len(ct_edges)-1

def cyl_chord(p0,p1,R,zc,hh):
    d=p1-p0; L=np.sqrt(np.sum(d*d,axis=1))
    dx=d[:,0]; dy=d[:,1]; dz=d[:,2]; x0=p0[:,0]; y0=p0[:,1]; z0=p0[:,2]
    a=dx*dx+dy*dy; b=x0*dx+y0*dy; c=x0*x0+y0*y0-R*R
    big=a>1e-16
    sq=np.sqrt(np.maximum(b*b-a*c,0))
    t1=(-b-sq)/np.maximum(a,1e-30); t2=(-b+sq)/np.maximum(a,1e-30)
    lo=np.minimum(t1,t2); hi=np.maximum(t1,t2)
    lo[~big]=-1e9; hi[~big]=1e9
    outside=(~big)&(x0*x0+y0*y0>=R*R)
    invdz=np.where(np.abs(dz)>1e-12,1.0/dz,0.0)
    z1=(zc-hh-z0)*invdz; z2=(zc+hh-z0)*invdz
    zlo=np.minimum(z1,z2); zhi=np.maximum(z1,z2)
    dz0=np.abs(dz)<1e-12
    zlo[dz0]=0.; zhi[dz0]=1.
    bad=dz0&(np.abs(z0-zc)>hh); zlo[bad]=1.; zhi[bad]=0.
    lo=np.maximum(np.maximum(lo,zlo),0); hi=np.minimum(np.minimum(hi,zhi),1)
    chord=L*np.maximum(0,hi-lo); chord[outside]=0
    return chord

# ================= photon cross sections =================
ph=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
Eph=ph[:,0]; coh=ph[:,1]; inc=ph[:,2]; pe=ph[:,3]; pn=ph[:,4]; pel=ph[:,5]; tot=ph[:,6]
Elg=np.logspace(np.log10(0.01),np.log10(2.0),2500)
def interp_log(x,y,xq):
    ok=(y>0)&(x>0); fi=interp1d(np.log(x[ok]),np.log(y[ok]),kind='linear',bounds_error=False,fill_value='extrapolate')
    return np.exp(fi(np.log(xq)))
COH=interp_log(Eph,coh,Elg); INC_i=interp_log(Eph,inc,Elg); PE=interp_log(Eph,pe,Elg)
PN0=interp_log(Eph,pn,Elg); PEL0=interp_log(Eph,pel,Elg); TOT=interp_log(Eph,tot,Elg)
logElg=np.log(Elg)
def xs_lookup(arr,E): return np.exp(np.interp(np.log(np.maximum(E,2e-3)),logElg,np.log(arr)))
def pair_xs(E):
    E=np.maximum(E,0); return np.where(E>1.022,xs_lookup(PN0,E)+xs_lookup(PEL0,E),0.0)
E1=1.1732; E2=1.3325
pe_edges=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])

def KN_weight(c,E):
    a=E/mec2; Ep=E/(1+a*(1-c)); rr=Ep/E
    return rr*rr*(rr+1/rr-(1-c*c))
def sample_KN_single(E,rng):
    fmax=KN_weight(-1.0,E)
    got=False
    while not got:
        c=rng.uniform(-1,1)
        if rng.uniform(0,1)*fmax < KN_weight(c,E):
            got=True
    Ep=E/(1+(E/mec2)*(1-c)); T=E-Ep
    pe_=np.sqrt(T*(T+2*mec2))
    cose=(E-Ep*c)/pe_ if pe_>0 else 1.0
    cose=np.clip(cose,-1,1)
    return T,cose,Ep,c

def sample_electron_state(Egamma, rng):
    winc=xs_lookup(INC_i,np.array([Egamma]))[0]; wpe=xs_lookup(PE,np.array([Egamma]))[0]; wpair=pair_xs(np.array([Egamma]))[0]
    W=winc+wpe+wpair
    r=rng.uniform(0,1)
    if r < winc/W:
        T,cose,Ep,c=sample_KN_single(Egamma,rng)
        return [('inc',T,cose)]
    elif r < (winc+wpe)/W:
        T=Egamma-0.0005
        cose=rng.uniform(0.75,1.0)
        return [('pe',T,cose)]
    else:
        Te=(Egamma-1.022)/2.0
        return [('pair',Te,0.6),('pair',Te,-0.6)]

# ================= source =================
def build_source_fns(photon_res):
    N=photon_res['N']; slab_prim=photon_res['slab_prim']; slab_sec=photon_res['slab_sec']
    zc=photon_res['zc_slabs']; hh=photon_res['hh_slab']; Vslab=np.pi*1.0*2*hh
    Omega105=109.97517633116931
    f1=slab_prim[0]/N/Vslab*Omega105; f2=slab_prim[1]/N/Vslab*Omega105
    fs=slab_sec/N/Vslab*Omega105
    return dict(zc=zc,f1=f1,f2=f2,fs=fs,
                z_fn1=interp1d(zc,f1,kind='linear',bounds_error=False,fill_value=(f1[0],f1[-1])),
                z_fn2=interp1d(zc,f2,kind='linear',bounds_error=False,fill_value=(f2[0],f2[-1])),
                fs_fn=lambda z: np.column_stack([np.interp(np.atleast_1d(z),zc,fs[:,k]) for k in range(28)]))

def sample_sources(z0,src,N_el,rng,R_lat=2.5,zr=1.2):
    zlo=max(z0-zr,0.01); zhi=min(z0+zr,7.9)
    zg=np.linspace(zlo,zhi,500)
    mu1=xs_lookup(TOT,np.array([E1]))[0]; mu2=xs_lookup(TOT,np.array([E2]))[0]
    f1=src['z_fn1'](zg); f2=src['z_fn2'](zg); fs=src['fs_fn'](zg)
    Ec=0.5*(pe_edges[:-1]+pe_edges[1:]); mut_sec=xs_lookup(TOT,Ec)
    S=(f1*mu1+f2*mu2)+fs@mut_sec
    dzg=zg[1]-zg[0]
    norm=np.trapz(S,zg)
    N_birth=np.pi*R_lat**2*norm
    w_el=N_birth/N_el
    cdf=np.concatenate([[0],np.cumsum(S*dzg)]); cdf/=cdf[-1]
    zsel=np.interp(rng.uniform(0,1,N_el),cdf,zg)
    r_l=rng.uniform(0,R_lat,N_el); phi_l=rng.uniform(0,2*np.pi,N_el)
    xl=r_l*np.cos(phi_l); yl=r_l*np.sin(phi_l)
    f1s=src['z_fn1'](zsel); f2s=src['z_fn2'](zsel); fs_sel=src['fs_fn'](zsel)
    w1=f1s*mu1; w2=f2s*mu2; ws=fs_sel*mut_sec[None,:]
    Wtot=w1+w2+ws.sum(axis=1)
    uu=rng.uniform(0,1,N_el)
    cw1=w1/Wtot; cw2=(w1+w2)/Wtot
    Egamma=np.empty(N_el); parent=np.zeros(N_el)
    mode=np.where(uu<cw1,0,np.where(uu<cw2,1,2))
    m0=mode==0; m1=mode==1; m2v=mode==2
    Egamma[m0]=E1; Egamma[m1]=E2
    cdfb=np.cumsum(ws,axis=1); cdfb=cdfb/cdfb[:,-1][:,None]
    bidx=np.clip(np.sum(rng.uniform(0,1,m2v.sum())[:,None]>cdfb[m2v],axis=1),0,27)
    Egamma[m2v]=Ec[bidx]; parent[m2v]=1
    E0=np.empty(N_el); cos0=np.empty(N_el); ispos=np.zeros(N_el,bool)
    for i in range(N_el):
        st=sample_electron_state(Egamma[i],rng)
        E0[i]=st[0][1]; cos0[i]=st[0][2]
        if len(st)>1:
            ispos[i]=True  # mark that a positron companion exists (we ignore it)
    az=rng.uniform(0,2*np.pi,N_el)
    sint=np.sqrt(np.maximum(0,1-cos0*cos0))
    pos=np.stack([xl,yl,100.0+zsel],axis=1)
    dd=np.stack([sint*np.cos(az),sint*np.sin(az),cos0],axis=1)
    dd/=np.linalg.norm(dd,axis=1,keepdims=True)
    return pos,dd,E0,parent,ispos,w_el,N_birth

# ================= transport =================
def rotate_e(d,th,ph):
    aux=np.broadcast_to(np.array([0.,0.,1.]),(len(d),3)).copy()
    aux[np.abs(d[:,2])>0.9]=np.array([1.,0.,0.])
    e1=np.cross(aux,d); n1=np.sqrt(np.sum(e1*e1,axis=1,keepdims=True)); e1=e1/np.maximum(n1,1e-12)
    e2=np.cross(d,e1)
    return np.cos(th)[:,None]*d+np.sin(th)[:,None]*(np.cos(ph)[:,None]*e1+np.sin(ph)[:,None]*e2)

def cos_delta_fn(T,E):
    return np.sqrt(np.maximum(0,T*(E+2*mec2)/(E*(T+2*mec2))))

def run_electron_mc(z0,N_el,seed,src,R_lat=2.5,zr=1.2):
    rng=np.random.default_rng(seed)
    pos,d,E0,parent,ispos,w_el,N_birth=sample_sources(z0,src,N_el,rng,R_lat=R_lat,zr=zr)
    maxN=N_el+8*N_el
    P=np.zeros((maxN,3)); Dd=np.zeros((maxN,3)); En=np.zeros(maxN)
    isdelta=np.zeros(maxN,bool); isposi=np.zeros(maxN,bool); Prnt=np.zeros(maxN); Al=np.zeros(maxN,bool)
    P[:N_el]=pos; Dd[:N_el]=d; En[:N_el]=E0; Prnt[:N_el]=parent; Al[:N_el]=True
    nactive=N_el; free=list(range(N_el,maxN))
    flu=np.zeros(nEB); ang=np.zeros(nCT); flu_delta=np.zeros(nEB)
    sumChord=0.0; sumE=0.0; sumScol=0.0; sumSrad=0.0; sumSrest=0.0
    trackend=0.0; dose_prim=0.0; dose_sec=0.0
    V=np.pi*0.2; zc=100.0+z0
    it=0; steps=0; n_delta_created=0; t0=time.time()
    while np.any(Al):
        idx=np.nonzero(Al)[0]
        if len(idx)==0: break
        Ei=En[idx]; St=Srest(Ei)+Srad_f(Ei)
        ds=np.maximum(np.minimum(0.02*Ei/np.maximum(St,1e-6),(Ei-Delta)/np.maximum(St,1e-6)),1e-6)
        p0=P[idx]; dd=Dd[idx]; p1=p0+dd*ds[:,None]
        ch=cyl_chord(p0,p1,1.0,zc,0.1)
        el_mask=~isposi[idx]
        if ch.sum()>0:
            ebin=np.searchsorted(eb_edges,Ei,side='right')-1
            cbin=np.searchsorted(ct_edges,dd[:,2],side='right')-1
            isd=isdelta[idx]
            for b in range(nEB):
                mm=(ebin==b)&(ch>0)
                if mm.any():
                    mm_e=mm&el_mask
                    flu[b]+=np.sum(ch[mm_e])
                    flu_delta[b]+=np.sum(ch[mm_e&isd])
            for b in range(nCT):
                mm=(cbin==b)&(ch>0)&el_mask
                if mm.any(): ang[b]+=np.sum(ch[mm])
            sumChord+=np.sum(ch[el_mask]); sumE+=np.sum(ch[el_mask]*Ei[el_mask])
            sumScol+=np.sum(ch[el_mask]*Scol_f(Ei[el_mask])); sumSrad+=np.sum(ch[el_mask]*Srad_f(Ei[el_mask]))
        Srest_i=Srest(Ei)
        sumSrest+=np.sum(ch*Srest_i)
        pr=Prnt[idx]
        dose_prim+=np.sum(ch[pr==0]*Srest_i[pr==0]); dose_sec+=np.sum(ch[pr==1]*Srest_i[pr==1])
        # delta
        lam=Ndel(Ei)*ds; nd=rng.poisson(lam); totd=int(nd.sum())
        if totd>0:
            if len(free)<totd+10:
                ext=max(totd+100,100000)
                P=np.vstack([P,np.zeros((ext,3))]); Dd=np.vstack([Dd,np.zeros((ext,3))])
                En=np.concatenate([En,np.zeros(ext)]); isdelta=np.concatenate([isdelta,np.zeros(ext,bool)])
                isposi=np.concatenate([isposi,np.zeros(ext,bool)]); Prnt=np.concatenate([Prnt,np.zeros(ext)]); Al=np.concatenate([Al,np.zeros(ext,bool)])
                free.extend(range(maxN,maxN+ext)); maxN+=ext
            reps=np.repeat(np.arange(len(idx)),nd)
            Td=sample_delta_energy(Ei[reps],len(reps),rng)
            cd=cos_delta_fn(Td,Ei[reps]); az_=rng.uniform(0,2*np.pi,len(reps))
            sd=np.sqrt(np.maximum(0,1-cd*cd)); d0=Dd[reps]
            aux=np.broadcast_to(np.array([0.,0.,1.]),(len(reps),3)).copy()
            aux[np.abs(d0[:,2])>0.9]=np.array([1.,0.,0.])
            e1=np.cross(aux,d0); n1=np.sqrt(np.sum(e1*e1,axis=1,keepdims=True)); e1=e1/np.maximum(n1,1e-12)
            e2=np.cross(d0,e1)
            dd_new=cd[:,None]*d0+sd[:,None]*(np.cos(az_)[:,None]*e1+np.sin(az_)[:,None]*e2)
            Esum=np.zeros(len(idx)); np.add.at(Esum,reps,Td)
            for j in range(len(reps)):
                fi=free.pop()
                P[fi]=p0[reps[j]]; Dd[fi]=dd_new[j]; En[fi]=Td[j]
                isdelta[fi]=True; Prnt[fi]=pr[reps[j]]; Al[fi]=True
                if fi>=nactive: nactive=fi+1
                n_delta_created+=1
        else:
            Esum=np.zeros(len(idx))
        sig=np.array([highland_sigma(e,ds[j]) for j,e in enumerate(Ei)])
        dth=rng.normal(0,sig); dph=rng.uniform(0,2*np.pi,len(idx))
        newd=rotate_e(dd,dth,dph)
        E_new=Ei-ds*(Srest(Ei)+Srad_f(Ei))-Esum
        P[idx]=p1; Dd[idx]=newd; En[idx]=E_new
        outside=(P[idx,2]<100.0)|(P[idx,2]>130.0)|(np.abs(P[idx,0])>15.0)|(np.abs(P[idx,1])>15.0)
        below=En[idx]<Delta
        in_vox=(np.abs(P[idx,2]-zc)<0.1)&(P[idx,0]**2+P[idx,1]**2<1.0)
        trackend+=np.sum(En[idx][below&in_vox])
        dead=outside|below
        Al[idx[dead]]=False
        it+=1; steps+=len(idx)
        if it>1000: break
    return dict(N_el=N_el,w_el=w_el,N_birth=N_birth,flu=flu,ang=ang,flu_delta=flu_delta,
                sumChord=sumChord,sumE=sumE,sumScol=sumScol,sumSrad=sumSrad,sumSrest=sumSrest,
                trackend=trackend,dose_prim=dose_prim,dose_sec=dose_sec,it=it,steps=steps,
                n_delta=n_delta_created,time=time.time()-t0)

if __name__=='__main__':
    rph=np.load('/tmp/photon_mc_final.npy',allow_pickle=True).item()
    src=build_source_fns(rph)
    res=run_electron_mc(5.0,20000,seed=101,src=src)
    print("time %.1f it %d steps %d ndelta %d"%(res['time'],res['it'],res['steps'],res['n_delta']))
    print("N_birth:",res['N_birth'],"w_el:",res['w_el'])
    V=np.pi*0.2
    print("e fluence/Phi0:",res['sumChord']/V*res['w_el'])
    D=(res['sumSrest']+res['trackend'])/V*res['w_el']*1.602e-10
    print("D/Phi0 Gycm2:",D)
    print("dose prim frac:",res['dose_prim']/(res['dose_prim']+res['dose_sec']))
    print("mean E:",res['sumE']/res['sumChord'])
    print("Srad/Scol:",res['sumSrad']/res['sumScol'])
    print("delta flu frac:",res['flu_delta'].sum()/res['flu'].sum())
    print("angular:",np.round(res['ang']/res['ang'].sum(),4))
    np.save('/tmp/emc_test.npy',res,allow_pickle=True)