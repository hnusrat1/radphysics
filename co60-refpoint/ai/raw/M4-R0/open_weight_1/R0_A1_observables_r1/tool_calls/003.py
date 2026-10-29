
import numpy as np, math, time, pickle

Etab = np.array([0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0],float)
mu_tab   = np.array([0.2269,0.2059,0.1838,0.1707,0.1505,0.1373,0.1186,0.1061,0.0969,0.0896,0.0787,0.0707,0.0632,0.0576,0.0494],float)
mu_coh   = np.array([0.0222,0.0193,0.0161,0.0154,0.0118,0.0097,0.0073,0.0061,0.0053,0.0047,0.0039,0.0033,0.0028,0.0024,0.0019],float)
mu_en_tab= np.array([0.0406,0.0323,0.0260,0.0256,0.0277,0.0297,0.0319,0.0328,0.0330,0.0329,0.0320,0.0309,0.0297,0.0285,0.0262],float)

def interp1(x, xe, ye):
    return float(np.exp(np.interp(np.log(x), np.log(xe), np.log(ye))))
def mu_total(E): return interp1(E,Etab,mu_tab)
def mu_re(E): return interp1(E,Etab,mu_coh)
def mu_en(E): return interp1(E,Etab,mu_en_tab)
def mu_pe(E): return 0.0064*(E/0.1)**(-3.2)
def mu_pair(E):
    if E<=1.022: return 0.0
    return 0.00025 + 0.00015*(E-1.25)
def comp_frac(E):
    m = mu_total(E); c = mu_re(E); p = mu_pe(E); q = mu_pair(E)
    comp = m - c - p - q
    if comp<0: comp=m*0.85
    return comp,m,c,p,q

rng = np.random.default_rng(12345)
def sample_compton_mu(alpha):
    while True:
        mu = rng.uniform(-1,1)
        r = 1.0/(1.0+alpha*(1.0-mu))
        f = r*r*(r + 1.0/r - (1.0 - mu*mu))
        if rng.uniform()*2.0 < f:
            return mu

def cyl_overlap(px,py,pz,dx,dy,dz,s,zc):
    lo=0.0; hi=s
    if abs(dz)<1e-12:
        if abs(pz-zc)>0.1+1e-9: return 0.0
    else:
        tz1=(zc-0.1-pz)/dz; tz2=(zc+0.1-pz)/dz
        a=min(tz1,tz2); b=max(tz1,tz2)
        lo=max(lo,a); hi=min(hi,b)
        if lo>=hi: return 0.0
    a=dx*dx+dy*dy
    if a<1e-14:
        if px*px+py*py>1+1e-9: return 0.0
    else:
        b=2*(px*dx+py*dy); c=px*px+py*py-1
        disc=b*b-4*a*c
        if disc<0:
            if px*px+py*py>1+1e-9: return 0.0
        else:
            sq=math.sqrt(disc)
            t1=(-b-sq)/(2*a); t2=(-b+sq)/(2*a)
            if c<=1e-12:
                hi=min(hi,t2)
            else:
                if t2<0 or t1>s: return 0.0
                lo=max(lo,t1); hi=min(hi,t2)
    return hi-lo if lo<hi else 0.0

edges_ph=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
edges_e=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
edges_c=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
nb_ph=28; nb_e=25; nb_c=10

M=2200000
zc=5.0; h=100.0; half=5.0; R=1.0; th=0.1
A=B=10.0; D=h
Om=4*math.atan((A*B/2.0)/(D*math.sqrt(4*D*D+A*A+B*B)))
V=math.pi*R*R*2*th
Cnorm=(Om/M)*(105.0)**2/V

depths=np.array([0.2,0.4,0.5,0.6,0.8,1.0,1.5,2.0,3.0,5.0,7.0,10.0])
Kdep=np.zeros(len(depths)); Kdep_p=np.zeros(len(depths)); Kdep_s=np.zeros(len(depths)); TLdep=np.zeros(len(depths))

tl_p=np.zeros(nb_ph); tl_s=np.zeros(nb_ph); tlp_c=np.zeros(nb_c); tls_c=np.zeros(nb_c)
sumE_p=0.0; sumE_s=0.0; sumLEmu_p=0.0; sumLEmu_s=0.0

t0=time.time()
for i in range(M):
    E = 1.1732 if rng.random()<0.5 else 1.3325
    x0=rng.uniform(-half,half); y0=rng.uniform(-half,half)
    lenv=math.sqrt(x0*x0+y0*y0+h*h)
    dx=x0/lenv; dy=y0/lenv; dz=h/lenv
    px=x0; py=y0; pz=0.0
    primary=True
    while True:
        m = mu_total(E)
        s = -math.log(rng.random())/m
        texit=1e30
        if dz>1e-12: tz1=(30.0-pz)/dz
        elif dz<-1e-12: tz1=(0.0-pz)/dz
        else: tz1=1e30
        if tz1<texit: texit=tz1
        if dx>1e-12: tx1=(15.0-px)/dx
        elif dx<-1e-12: tx1=(-15.0-px)/dx
        else: tx1=1e30
        if tx1<texit: texit=tx1
        if dy>1e-12: ty1=(15.0-py)/dy
        elif dy<-1e-12: ty1=(-15.0-py)/dy
        else: ty1=1e30
        if ty1<texit: texit=ty1
        if texit<=0: break
        seg = s if s<texit else texit
        # main cylinder at zc
        L = cyl_overlap(px,py,pz,dx,dy,dz,seg,zc)
        if L>0:
            ebin=int(np.searchsorted(edges_ph,E,side='right')-1); cbin=int(np.searchsorted(edges_c,dz,side='right')-1)
            if primary:
                tl_p[ebin]+=L; tlp_c[cbin]+=L; sumE_p+=L*E; sumLEmu_p+=L*E*mu_en(E)
            else:
                tl_s[ebin]+=L; tls_c[cbin]+=L; sumE_s+=L*E; sumLEmu_s+=L*E*mu_en(E)
        for k in range(len(depths)):
            Ld=cyl_overlap(px,py,pz,dx,dy,dz,seg,depths[k])
            if Ld>0:
                emu=Ld*E*mu_en(E)
                Kdep[k]+=emu
                if primary: Kdep_p[k]+=emu
                else: Kdep_s[k]+=emu
                TLdep[k]+=Ld
        if s>texit: break
        pxi=px+dx*s; pyi=py+dy*s; pzi=pz+dz*s
        comp,m,c,p,q=comp_frac(E)
        tot=comp+c+p+q
        r=rng.random()*tot
        if r<comp:
            alpha=E/0.511
            mu=sample_compton_mu(alpha)
            Ep=E/(1.0+alpha*(1.0-mu))
            # rotate direction
            phi=rng.uniform(0,2*math.pi)
            cp=math.sqrt(max(0.0,1.0-mu*mu))
            zdirx=dx; zdiry=dy; zdirz=dz
            if abs(dz)<0.99: ux=1.0; uy=0.0; uz=0.0
            else: ux=0.0; uy=1.0; uz=0.0
            dot=ux*zdirx+uy*zdiry+uz*zdirz
            ux-=dot*zdirx; uy-=dot*zdiry; uz-=dot*zdirz
            nu=math.sqrt(ux*ux+uy*uy+uz*uz)
            ux/=nu; uy/=nu; uz/=nu
            vx=zdiry*uz-zdirz*uy; vy=zdirz*ux-zdirx*uz; vz=zdirx*uy-zdiry*ux
            sp=math.sin(phi); co=math.cos(phi)
            ndx=mu*zdirx+sp*cp*ux+co*cp*vx
            ndy=mu*zdiry+sp*cp*uy+co*cp*vy
            ndz=mu*zdirz+sp*cp*uz+co*cp*vz
            dx=ndx; dy=ndy; dz=ndz
            E=Ep
            px=pxi; py=pyi; pz=pzi
            primary=False
        elif r<comp+c:
            px=pxi; py=pyi; pz=pzi
            primary=False
        elif r<comp+c+p:
            break  # photoelectric absorbed
        else:
            break  # pair: photon gone (annihilation ignored)
print("time",time.time()-t0,"M",M)

C = Cnorm
prim_per_phi0 = C*np.sum(tl_p)
tot_per_phi0 = C*(np.sum(tl_p)+np.sum(tl_s))
print("primary/Phi0",prim_per_phi0,"scattered TL sum",tl_s.sum(),"total/Phi0",tot_per_phi0)
print("scattered fluence/Phi0",C*tl_s.sum())
print("scattered fraction fluence",tl_s.sum()/(tl_p.sum()+tl_s.sum()))
print("scattered fraction energy fluence",sumE_s/(sumE_p+sumE_s))
print("mean photon energy",(sumE_p+sumE_s)/(tl_p.sum()+tl_s.sum()))
print("mean primary energy",sumE_p/tl_p.sum(),"mean scattered",sumE_s/tl_s.sum())
print("angular primary",tlp_c/tlp_c.sum())
print("angular scattered",tls_c/tls_c.sum())
print("Kcol/Phi0 Gy cm2",1.602e-10*C*(sumLEmu_p+sumLEmu_s))
print("Kcol prim,scatt share",1.602e-10*C*sumLEmu_p,1.602e-10*C*sumLEmu_s)
print("primary spectrum/Phi0", C*tl_p)
print("scattered spectrum/Phi0", C*tl_s)
print("depths",depths)
Kcol_d=1.602e-10*C*Kdep
print("Kcol by depth",Kcol_d)
print("PDD5",100*Kcol_d[depths==5.0][0]/Kcol_d.max())
print("scatter dose frac by depth",Kdep_s/Kdep)

# save
np.savez('mc_results.npz', tl_p=tl_p, tl_s=tl_s, tlp_c=tlp_c, tls_c=tls_c,
         sumE_p=sumE_p, sumE_s=sumE_s, sumLEmu_p=sumLEmu_p, sumLEmu_s=sumLEmu_s,
         Kdep=Kdep, Kdep_p=Kdep_p, Kdep_s=Kdep_s, TLdep=TLdep, depths=depths)
