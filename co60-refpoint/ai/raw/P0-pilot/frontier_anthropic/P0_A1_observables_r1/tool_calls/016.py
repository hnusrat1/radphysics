
ana = r'''
import sys, pickle, numpy as np
sys.path.insert(0,'/tmp/work')
from phys import *
trap=np.trapezoid
outs=[pickle.load(open(f'/tmp/work/out_{s}.pkl','rb')) for s in [11,12,13,14]]
Nt=sum(o['N'] for o in outs)
Omega=outs[0]['Omega']; Vc=outs[0]['Vc']
NORM=Omega*110.0**2/(Nt*Vc)
inter={k:sum(o['inter'][k] for o in outs) for k in outs[0]['inter']}
transf={k:sum(o['transf'][k] for o in outs) for k in outs[0]['transf']}
Lsum={p:sum(o['Lsum'][p] for o in outs) for p in 'PS'}
ESum={p:sum(o['ESum'][p] for o in outs) for p in 'PS'}
hp=sum(o['hist_prim'] for o in outs); hs=sum(o['hist_scat'] for o in outs)
ap=sum(o['ang_prim'] for o in outs); as_=sum(o['ang_scat'] for o in outs)
ep={k:np.concatenate([o['ep'][k] for o in outs]) for k in outs[0]['ep']}
ev={k:np.concatenate([o['ev'][k] for o in outs]) for k in outs[0]['ev']}
names={'c':'compton','pe':'photoelectric','pair':'pair','coh':'rayleigh'}
def fr(d,ps):
    tot=sum(d[(p,t)] for p in ps for t in names)
    return {names[t]: float(sum(d[(p,t)] for p in ps)/tot) for t in names}
R={}
fP=NORM*Lsum['P']; fS=NORM*Lsum['S']
R['L1_primary_fluence_per_phi0']=fP
R['L1_total_fluence_per_phi0']=fP+fS
R['L1_scattered_fraction_fluence']=fS/(fP+fS)
R['L1_scattered_fraction_energy_fluence']=ESum['S']/(ESum['P']+ESum['S'])
R['L1_mean_photon_energy_MeV']=(ESum['P']+ESum['S'])/(Lsum['P']+Lsum['S'])
R['L1_photon_fluence_spectrum_primary']=[0.0]*28
R['L1_photon_fluence_spectrum_scattered']=list(NORM*hs)
R['L1_angular_primary']=list(ap/ap.sum())
R['L1_angular_scattered']=list(as_/as_.sum())
R['L2_interaction_fractions']=fr(inter,'PS')
R['L2_energy_transfer_fractions']=fr(transf,'PS')
R['L2_interaction_fractions_primary']=fr(inter,'P')
R['L2_interaction_fractions_scattered']=fr(inter,'S')
R['L2_scattered_share_of_interactions']=float(sum(inter[('S',t)] for t in names)/sum(inter.values()))
# ---- Layer 3 ----
edges3=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
cb=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
def hist_e(T,w): 
    h,_=np.histogram(T,bins=edges3,weights=w); return h
w=ep['w']; T=ep['T']; pr=ep['prim']; cz=ep['cos']; kind=ep['kind']
h3=hist_e(T,w); R['L3_initial_spectrum']=list(h3/h3.sum())
hp3=hist_e(T[pr],w[pr]); R['L3_initial_spectrum_parent_primary']=list(hp3/hp3.sum())
R['L3_mean_initial_energy_MeV']=float((w*T).sum()/w.sum())
R['L3_mean_initial_energy_parent_primary_MeV']=float((w*T)[pr].sum()/w[pr].sum())
hd,_=np.histogram(np.clip(cz,-1,1),bins=cb,weights=w); R['L3_initial_direction']=list(hd/hd.sum())
# ---- Layer 4 : CSDA-type recursion with Moller delta rays ----
dE=0.002; n=1000; Eg=(np.arange(n)+0.5)*dE
Qg=np.zeros(n)
m4=kind!=3   # electrons only
hq,_=np.histogram(T[m4],bins=n,range=(0,n*dE),weights=w[m4]); Qg=NORM*hq/dE  # per MeV per Phi0
def Sbethe(Tk, ZA=0.55508, I=75e-6):
    t=Tk/mc2; b2=t*(t+2)/(t+1)**2
    F=(1-b2)*(1+t*t/8-(2*t+1)*np.log(2))
    return 0.153537*ZA/b2*(np.log(t*t*(t+2)/(2*(I/mc2)**2))+F)
def Stot(Tk): return Sbethe(Tk)*(1+0.00927*Tk)
def dsigdT(Tp,Tk):
    # Moller differential cross section per electron, per MeV of delta energy (cm2/MeV)
    t=Tk/mc2; b2=t*(t+2)/(t+1)**2
    eps=Tp/Tk
    f=1/eps**2+1/(1-eps)**2+(t/(t+1))**2-(2*t+1)/(t+1)**2/(eps*(1-eps))
    return 2*np.pi*re**2*mc2/(b2*Tk)*f/Tk
def recursion(withdelta):
    Phi=np.zeros(n); cum=0.0
    Qd=np.zeros(n)
    for i in range(n-1,4,-1):
        Ti=Eg[i]
        if withdelta:
            k=np.where(Eg>=2*Ti)[0]
            if k.size:
                Qd[i]=ne*np.sum(Phi[k]*dE*dsigdT(Ti,Eg[k]))
        Qi=Qg[i]+Qd[i]
        cum+=Qi*dE
        Phi[i]=cum/Stot(Ti)
    return Phi
Phi_t=recursion(True); Phi_n=recursion(False)
sel=np.arange(n)>=5
L4_fl=np.sum(Phi_t[sel]*dE)
R['L4_electron_fluence_per_phi0']=float(L4_fl)
spec=[]
for j in range(len(edges3)-1):
    m=(Eg>=edges3[j])&(Eg<edges3[j+1])
    spec.append(float(np.sum(Phi_t[m]*dE)))
R['L4_electron_fluence_spectrum']=spec
R['L4_delta_fraction']=float(1-np.sum(Phi_n[sel]*dE)/L4_fl)
R['L4_mean_electron_energy_MeV']=float(np.sum(Eg[sel]*Phi_t[sel]*dE)/L4_fl)
# angular: approximate with production direction distribution of electrons (L3 all-electron)
R['L4_angular']=list(hd/hd.sum())
Srad_ratio=lambda Tk: 0.00927*Tk
R['L4_radiative_to_collision']=float(np.sum(Phi_t[sel]*dE*Sbethe(Eg[sel])*Srad_ratio(Eg[sel]))/np.sum(Phi_t[sel]*dE*Sbethe(Eg[sel])))
# ---- Layer 5 ----
conv=1.602176e-10
Ktot=NORM*sum(transf.values())*conv
g=0.004635*np.sum(w*T**2)/np.sum(w*T)
Kcol=Ktot*(1-g)
R['L5_K_over_phi0_Gy_cm2']=float(Ktot)
R['L5_Kcol_over_phi0_Gy_cm2']=float(Kcol)
R['L5_g']=float(g)
R['L5_D_over_phi0_Gy_cm2']=float(Kcol)   # CPE
R['L5_beta']=1.0
R['L5_dose_fraction_parent_scattered']=float(sum(transf[('S',t)] for t in names)/sum(transf.values()))
# Spencer-Attix water/air, Delta=10 keV
def Lres(Tk,ZA,I,nelec):
    S=Sbethe(Tk,ZA,I)
    # delta energy above Delta: nelec * int_Delta^{T/2} Tp dsig/dTp dTp
    Dl=0.01
    Tp=np.linspace(Dl,Tk/2,400)
    out=np.zeros_like(Tk)
    for j,tk in enumerate(np.atleast_1d(Tk)):
        if tk/2<=Dl: continue
        tp=np.linspace(Dl,tk/2,400)
        out[j]=nelec*trap(tp*dsigdT(tp,tk),tp)
    return S-out
Esel=Eg[sel]
Lw=np.array([Sbethe(e)-(lambda tk:(0 if tk/2<=0.01 else ne*trap(np.linspace(0.01,tk/2,400)*dsigdT(np.linspace(0.01,tk/2,400),tk),np.linspace(0.01,tk/2,400))))(e) for e in Esel])
La=np.array([Sbethe(e,0.49919,85.7e-6)-(0 if e/2<=0.01 else (6.022e23*0.49919)*trap(np.linspace(0.01,e/2,400)*dsigdT(np.linspace(0.01,e/2,400),e),np.linspace(0.01,e/2,400))) for e in Esel])
R['L5_s_w_air']=float(np.sum(Phi_t[sel]*Lw)/np.sum(Phi_t[sel]*La))
# ---- depth dose (straight-ahead CSDA tally) ----
Tg=np.linspace(1e-5,2.0,40001)
Sg=Sbethe(Tg)
Rg=np.concatenate([[0],np.cumsum(0.5*(1/Sg[1:]+1/Sg[:-1])*np.diff(Tg))])
def Rof(E): return np.interp(E,Tg,Rg)
x=ev['x']; Te=ev['T']; uu=ev['u']
z0=x[:,2]; uz=uu[:,2]
edz=np.arange(0,30.01,0.1); D=np.zeros(len(edz)-1)
M=16
Rt=Rof(Te)
for j in range(M):
    Ea=Te*(1-j/M); Eb_=Te*(1-(j+1)/M)
    sa=Rt-Rof(Ea); sb=Rt-Rof(Eb_)
    smid=0.5*(sa+sb)
    zz=z0+uz*smid
    dEj=(Ea-Eb_)*0  # placeholder
    dEj=Te/M
    m=(zz>=0)&(zz<30)
    h,_=np.histogram(zz[m],bins=edz,weights=dEj[m])
    D+=h
zc=0.5*(edz[1:]+edz[:-1])
imax=np.argmax(D[:]); 
Dmax=D[imax]
i5=np.argmin(abs(zc-5.0))
R['L5_PDD5_percent']=float(100*D[i5]/Dmax)
R['_dmax_cm']=float(zc[imax]); R['_PDD']=list(map(float,100*D/Dmax)[:60])
R['_Kcol_check_pointcount']=float(len(Te))
pickle.dump(R,open('/tmp/work/R.pkl','wb'))
import json
print(json.dumps({k:v for k,v in R.items() if not k.startswith('_PDD')},indent=1,default=float))
print('dmax', zc[imax], 'PDD at depths 0.5..10:', [round(float(100*D[np.argmin(abs(zc-z))]/Dmax),1) for z in [0.5,1,1.5,2,3,4,5,6,8,10]])
'''
open('/tmp/work/ana.py','w').write(ana)
import subprocess
r=subprocess.run(['python','/tmp/work/ana.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-6000:],r.stderr[-3000:])
