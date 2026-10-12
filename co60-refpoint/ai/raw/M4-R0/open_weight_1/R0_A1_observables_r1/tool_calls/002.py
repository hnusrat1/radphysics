
import numpy as np, math, time

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

# Compton sampling: returns cos(photon scatter angle)
def sample_compton_mu(alpha):
    # alpha = E/0.511
    # f(mu) = (E'/E)^2 [E'/E + E/E' - (1-mu^2)]  with E'/E = 1/(1+alpha(1-mu))
    while True:
        mu = np.random.uniform(-1,1)
        r = 1.0/(1.0+alpha*(1.0-mu))
        f = r*r*(r + 1.0/r - (1.0 - mu*mu))
        if np.random.uniform()*2.0 < f:  # f_max=2 at mu=1
            return mu

def ray_cylinder_overlap(p0, d, s, zc):
    """track length inside cylinder radius1, z in [zc-0.1,zc+0.1] for segment [0,s]"""
    z0=p0[2]; dz=d[2]
    lo=0.0; hi=s
    if abs(dz)<1e-12:
        if abs(z0-zc)>0.1+1e-9: return 0.0
    else:
        tz1=(zc-0.1-z0)/dz; tz2=(zc+0.1-z0)/dz
        a=min(tz1,tz2); b=max(tz1,tz2)
        lo=max(lo,a); hi=min(hi,b)
        if lo>=hi: return 0.0
    px=p0[0]; py=p0[1]; dx=d[0]; dy=d[1]
    a=dx*dx+dy*dy
    if a<1e-14:
        if px*px+py*py>1+1e-9: return 0.0
    else:
        b=2*(px*dx+py*dy); c=px*px+py*py-1
        disc=b*b-4*a*c
        if disc<0:
            if px*px+py*py>1+1e-9: return 0.0
            # start inside? disc<0 with c<0 => always inside
        else:
            sq=math.sqrt(disc)
            t1=(-b-sq)/(2*a); t2=(-b+sq)/(2*a)
            if c<=1e-12:  # starts inside
                lo=max(lo,0.0); hi=min(hi,t2)
            else:
                if t2<0 or t1>s: return 0.0
                lo=max(lo,t1); hi=min(hi,t2)
    if lo<hi: return hi-lo
    return 0.0

# bins
edges_ph=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
edges_e=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
edges_c=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
nb_ph=len(edges_ph)-1; nb_e=len(edges_e)-1; nb_c=len(edges_c)-1

def binidx(val, edges):
    # value within [edges[0], edges[-1]]
    i = int(np.searchsorted(edges, val, side='right')-1)
    if i<0: i=0
    if i>=len(edges)-1: i=len(edges)-2
    return i

M=400000
zc=5.0
# geometry
h=100.0; half=5.0; R=1.0; th=0.1  # cylinder half-thickness
# pyramid solid angle
A=B=10.0; D=100.0
Om=4*math.atan((A*B/2.0)/(D*math.sqrt(4*D*D+A*A+B*B)))
V=math.pi*R*R*2*th
Cnorm=(Om/M)*(D+5.0)**2/V  # note: 105 cm from source to POI

depths=np.array([0.2,0.4,0.5,0.6,0.8,1.0,1.5,2.0,3.0,5.0,7.0,10.0])
Kdep=np.zeros(len(depths)); Kdep_p=np.zeros(len(depths)); Kdep_s=np.zeros(len(depths))

tl_p=np.zeros(nb_ph); tl_s=np.zeros(nb_ph)
tlp_c=np.zeros(nb_c); tls_c=np.zeros(nb_c)
sumE_p=0.0; sumE_s=0.0
sumLEmu_p=0.0; sumLEmu_s=0.0
intc_p={'comp':0,'pe':0,'pair':0,'ray':0}
intc_s={'comp':0,'pe':0,'pair':0,'ray':0}
trans={'comp':0.0,'pe':0.0,'pair':0.0,'ray':0.0}
# initial electrons
Te_all=[]; Te_prim=[]; cos_all=[]; cos_prim=[]
nint=0; nint_sc=0

t0=time.time()
for i in range(M):
    E = 1.1732 if np.random.rand()<0.5 else 1.3325
    x0=np.random.uniform(-half,half); y0=np.random.uniform(-half,half)
    lenv=math.sqrt(x0*x0+y0*y0+h*h)
    dx=x0/lenv; dy=y0/lenv; dz=h/lenv
    px=x0; py=y0; pz=0.0
    primary=True
    while True:
        m = mu_total(E)
        s = -math.log(np.random.rand())/m
        # exit distances
        texit=1e30
        if dz>1e-12: texit=min(texit,(30.0-pz)/dz)
        elif dz<-1e-12: texit=min(texit,(0.0-pz)/dz)
        if dx>1e-12: texit=min(texit,(15.0-px)/dx)
        elif dx<-1e-12: texit=min(texit,(-15.0-px)/dx)
        if dy>1e-12: texit=min(texit,(15.0-py)/dy)
        elif dy<-1e-12: texit=min(texit,(-15.0-py)/dy)
        if texit<=0: break
        seg=min(s,texit)
        # score at main cylinder zc=5
        L=ray_cylinder_overlap((px,py,pz),(dx,dy,dz),seg,zc)
        if L>0:
            ebin=binidx(E,edges_ph); cbin=binidx(dz,edges_c)
            if primary:
                tl_p[ebin]+=L; tlp_c[cbin]+=L; sumE_p+=L*E; sumLEmu_p+=L*E*mu_en(E)
            else:
                tl_s[ebin]+=L; tls_c[cbin]+=L; sumE_s+=L*E; sumLEmu_s+=L*E*mu_en(E)
        # depth-dose cylinders
        for k,zc2 in enumerate(depths):
            Ld=ray_cylinder_overlap((px,py,pz),(dx,dy,dz),seg,zc2)
            if Ld>0:
                Kdep[k]+=Ld*E*mu_en(E)
                if primary: Kdep_p[k]+=Ld*E*mu_en(E)
                else: Kdep_s[k]+=Ld*E*mu_en(E)
        if s>texit:  # escaped
            break
        # interaction
        pxi=px+dx*s; pyi=py+dy*s; pzi=pz+dz*s
        comp,m,c,p,q=comp_frac(E)
        tot=comp+c+p+q
        r=np.random.rand()*tot
        inside = (abs(pzi-zc)<=th) and (pxi*pxi+pyi*pyi<=R*R)
        if r<comp:
            if inside:
                alpha=E/0.511
                mu= sample_compton_mu(alpha)
                Ep=E/(1.0+alpha*(1.0-mu))
                Te=E-Ep
                pe=math.sqrt(E*E+Ep*Ep-2.0*E*Ep*mu)
                cose=(E-Ep*mu)/pe if pe>0 else 1.0
                intc_p['comp']+= 1 if primary else 0
                intc_s['comp']+= 0 if primary else 1
                trans['comp']+=Te
                Te_all.append(Te); cos_all.append(cose)
                if primary: Te_prim.append(Te); cos_prim.append(cose)
                nint+=1; nint_sc+= 0 if primary else 1
                E=Ep
                # new direction
                # rotation: choose azimuth uniform
                phi=np.random.uniform(0,2*math.pi)
                # direction unit
                cp = math.sqrt(max(0.0,1.0-mu*mu))
                # rotate around z
                dperp_x = dx  # new perp component: use existing dir? simpler: build from beam axis
                # We'll construct using current dir as z-axis approximation; to keep simple use vector in plane
                # Actually need precise: new dir components in original coord.
                zdir = (dx,dy,dz)
                # pick arbitrary perpendicular u
                ux,uy,uz = (1,0,0) if abs(dz)<0.99 else (0,1,0)
                # u -= (u.z) zdir
                dot = ux*zdir[0]+uy*zdir[1]+uz*zdir[2]
                ux-=dot*zdir[0]; uy-=dot*zdir[1]; uz-=dot*zdir[2]
                nu=math.sqrt(ux*ux+uy*uy+uz*uz)
                ux/=nu; uy/=nu; uz/=nu
                # v = zdir x u
                vx=zdir[1]*uz-zdir[2]*uy; vy=zdir[2]*ux-zdir[0]*uz; vz=zdir[0]*uy-zdir[1]*ux
                # new = mu*zdir + cp sin(phi) u + cp cos(phi) v ... standard
                sp=math.sin(phi); co=math.cos(phi)
                ndx=mu*zdir[0]+sp*cp*ux+co*cp*vx
                ndy=mu*zdir[1]+sp*cp*uy+co*cp*vy
                ndz=mu*zdir[2]+sp*cp*uz+co*cp*vz
                dx,dy,dz=ndx,ndy,ndz
                px,py,pz=pxi,pyi,pzi
                primary=False
                continue
            else:
                # scatter no scoring
                alpha=E/0.511
                mu= sample_compton_mu(alpha)
                Ep=E/(1.0+alpha*(1.0-mu))
                E=Ep
                phi=np.random.uniform(0,2*math.pi)
                cp = math.sqrt(max(0.0,1.0-mu*mu))
                zdir=(dx,dy,dz)
                ux,uy,uz = (1,0,0) if abs(dz)<0.99 else (0,1,0)
                dot=ux*zdir[0]+uy*zdir[1]+uz*zdir[2]
                ux-=dot*zdir[0]; uy-=dot*zdir[1]; uz-=dot*zdir[2]
                nu=math.sqrt(ux*ux+uy*uy+uz*uz); ux/=nu; uy/=nu; uz/=nu
                vx=zdir[1]*uz-zdir[2]*uy; vy=zdir[2]*ux-zdir[0]*uz; vz=zdir[0]*uy-zdir[1]*ux
                sp=math.sin(phi); co=math.cos(phi)
                ndx=mu*zdir[0]+sp*cp*ux+co*cp*vx
                ndy=mu*zdir[1]+sp*cp*uy+co*cp*vy
                ndz=mu*zdir[2]+sp*cp*uz+co*cp*vz
                dx,dy,dz=ndx,ndy,ndz
                px,py,pz=pxi,pyi,pzi
                primary=False
                continue
        elif r<comp+c:
            # Rayleigh: elastic, forward
            if inside:
                intc_p['ray']+= 1 if primary else 0
                intc_s['ray']+= 0 if primary else 1
                trans['ray']+=0.0
                nint+=1; nint_sc+= 0 if primary else 1
            px,py,pz=pxi,pyi,pzi
            # keep direction (approximately forward)
            primary=False
            continue
        elif r<comp+c+p:
            # photoelectric
            if inside:
                Te=E-0.0005
                intc_p['pe']+= 1 if primary else 0
                intc_s['pe']+= 0 if primary else 1
                trans['pe']+=E
                Te_all.append(Te); cos_all.append(1.0)
                if primary: Te_prim.append(Te); cos_prim.append(1.0)
                nint+=1; nint_sc+= 0 if primary else 1
            break
        else:
            # pair
            if inside:
                intc_p['pair']+= 1 if primary else 0
                intc_s['pair']+= 0 if primary else 1
                trans['pair']+=E-1.022
                Te1=(E-1.022)/2.0; Te2=Te1
                Te_all.append(Te1); cos_all.append(1.0)
                Te_all.append(Te2); cos_all.append(1.0)
                if primary:
                    Te_prim.append(Te1); cos_prim.append(1.0)
                    Te_prim.append(Te2); cos_prim.append(1.0)
                nint+=1; nint_sc+= 0 if primary else 1
            break

print("time",time.time()-t0,"for",M)
print("nint",nint,"nint_sc",nint_sc)
# normalize
C = Cnorm  # (Om/M)*(105^2)/V
Phi_prim = C*np.sum(tl_p)
print("primary/Phi0",Phi_prim, "exp(-mu5) analytic ~", 0.5*(math.exp(-5*interp1(1.1732,Etab,mu_tab))+math.exp(-5*interp1(1.3325,Etab,mu_tab))))
print("tl_p sum",tl_p.sum(),"tl_s sum",tl_s.sum())
tl_tot=tl_p+tl_s
print("total/Phi0",C*tl_tot.sum(),"scattered fraction fluence", tl_s.sum()/tl_tot.sum())
print("scattered fraction energy fluence", sumE_s/(sumE_p+sumE_s))
print("mean photon energy MeV", (sumE_p+sumE_s)/(tl_p.sum()+tl_s.sum()))
print("angular primary",tlp_c/tlp_c.sum())
print("angular scattered",tls_c/tls_c.sum())
print("interactions",intc_p,intc_s)
print("L2_scattered_share", nint_sc/nint)
totc={k:intc_p[k]+intc_s[k] for k in intc_p}
totall=sum(totc.values())
print("L2 fractions", {k:v/totall for k,v in totc.items()})
tt=trans
tsum=sum(tt.values())
print("L2 transfer fractions", {k:v/tsum for k,v in tt.items()})
print("L2 primary fractions", {k:intc_p[k]/sum(intc_p.values()) for k in intc_p})
print("L2 scattered fractions", {k:intc_s[k]/sum(intc_s.values()) for k in intc_s})
print("Kcol/Phi0 Gy cm2", 1.602e-10*C*(sumLEmu_p+sumLEmu_s))
print("Kcol primary/Phi0",1.602e-10*C*sumLEmu_p, "scattered",1.602e-10*C*sumLEmu_s)
# L3
Te_all=np.array(Te_all); Te_prim=np.array(Te_prim); cos_all=np.array(cos_all); cos_prim=np.array(cos_prim)
hist_all=np.histogram(Te_all,bins=edges_e)[0]
hist_prim=np.histogram(Te_prim,bins=edges_e)[0]
print("L3 initial spectrum (frac)",hist_all/hist_all.sum())
print("L3 parent primary",hist_prim/hist_prim.sum())
print("mean initial",Te_all.mean(),"parent primary",Te_prim.mean())
print("L3 initial direction",np.histogram(cos_all,bins=edges_c)[0]/len(cos_all))
print("Kdep raw",Kdep)
