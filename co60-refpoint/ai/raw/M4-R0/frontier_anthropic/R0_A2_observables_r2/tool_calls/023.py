
ana4 = r'''
import sys, pickle, numpy as np
sys.path.insert(0,'/tmp')
import phys, mc as M
from phys import mu_inc, mu_pe, mu_pair, mu_coh, mu_tot_fn, mec2, re, NeW, NeAir, es
t=pickle.load(open('/tmp/tallies.pkl','rb'))
N=t['N']; V=np.pi*1.0**2*0.2; Vslab=np.pi*1.0**2*0.1
Om=M.OMEGA; r0=105.0
Fn=Om*r0**2/(N*V); Fslab=Om*r0**2/(N*Vslab)
CG=1.602177e-10
ELINES=np.array([1.1732,1.3325]); WL=np.array([0.5,0.5])
def prim_line_ratio(E):
    mu=float(mu_tot_fn(np.array([E]))[0])
    zz=np.linspace(4.9,5.1,2001)
    f=np.exp(-mu*zz)*(r0/(100+zz))**2
    return np.trapezoid(f,zz)/0.2
R_line=np.array([0.5*prim_line_ratio(E) for E in ELINES])
EC=(np.arange(M.NEB)+0.5)*M.DE
Rs_fine=t['fl_E'][1]*Fn
edges28=np.arange(0,1.4001,0.05)
def rebin(fine, edges_coarse):
    idx=np.round(edges_coarse/M.DE).astype(int)
    return np.array([fine[idx[i]:idx[i+1]].sum() for i in range(len(edges_coarse)-1)])
prim_spec=np.zeros(28)
for E,R in zip(ELINES,R_line): prim_spec[int(np.floor(E/0.05))]+=R
scat_spec=rebin(Rs_fine,edges28)
tot_spec=prim_spec+scat_spec
Phi_prim=R_line.sum(); Phi_scat=Rs_fine.sum(); Phi_tot=Phi_prim+Phi_scat
L1_scat_frac=Phi_scat/Phi_tot
en_scat=(Rs_fine*EC).sum(); en_prim=(R_line*ELINES).sum()
L1_scat_energy_frac=en_scat/(en_scat+en_prim)
L1_mean_E=(en_scat+en_prim)/Phi_tot
angP=np.zeros(10); angP[9]=1.0
angS=t['fl_cos'][1]/t['fl_cos'][1].sum()
Wi=t['W_int']; We=t['W_etr']
names=['compton','photoelectric','pair','rayleigh']
def fr4(vec):
    v=np.array([vec[1],vec[2],vec[3],vec[0]]); v=v/v.sum()
    return dict(zip(names,[float(x) for x in v]))
L2=fr4(Wi.sum(0)); L2e=fr4(We.sum(0)); L2p=fr4(Wi[0]); L2s=fr4(Wi[1])
L2_share=Wi[1].sum()/Wi.sum()
rec=t['recs']; T_e=rec['T']; c_e=rec['cos']; w_e=rec['w']; f_e=rec['flag']
edges25=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
cedges=np.linspace(-1,1,11)
def hist(x,w,edges):
    h,_=np.histogram(x,bins=edges,weights=w); return h
h3=hist(T_e,w_e,edges25); L3_spec=h3/h3.sum()
m0=f_e==0
h3p=hist(T_e[m0],w_e[m0],edges25); L3_spec_p=h3p/h3p.sum()
L3_mean=(T_e*w_e).sum()/w_e.sum(); L3_mean_p=(T_e[m0]*w_e[m0]).sum()/w_e[m0].sum()
hc=hist(np.clip(c_e,-1,1-1e-12),w_e,cedges); L3_dir=hc/hc.sum()
# ---------- electron grid ----------
DT=0.005
Tedges=np.arange(0,1.4+1e-9,DT); Tcen=(Tedges[:-1]+Tedges[1:])/2; nT=len(Tcen)
i10=int(round(0.01/DT))
def kn_hist(E):
    g,w=np.polynomial.legendre.leggauss(3000)
    k=E/mec2; e=1/(1+k*(1-g))
    dsig=e*e*(e+1/e-(1-g*g))*w
    T=E*(1-e)
    h,_=np.histogram(T,bins=Tedges,weights=dsig)
    return h/h.sum()
Pmat=np.zeros((M.NEB,nT))
for j in range(M.NEB):
    if EC[j]>0.0011: Pmat[j]=kn_hist(EC[j])
# stopping powers on T grid (T>=10 keV used)
Scol=np.zeros(nT); Srad=np.zeros(nT); dens=np.zeros(nT)
ok=Tcen>=0.0099
ev=np.array([phys.estar(T) for T in Tcen[ok]])   # cols: Scol,Srad,Stot,CSDA,Yrad,delta
Scol[ok]=ev[:,0]; Srad[ok]=ev[:,1]; dens[ok]=ev[:,5]
DELTA=0.01
tail_w=np.zeros(nT); tail_w[ok]=phys.tail_fast(Tcen[ok],DELTA,NeW)
Lw=Scol-tail_w
Lw_tot=Lw+Srad
def bethe_B(T,I,delta):
    tau=T/mec2; b2=phys.beta2(T)
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
    return np.log(tau**2*(tau+2)/(2*(I/mec2)**2))+F-delta
I_w=75e-6; I_a=85.7e-6
K0=2*np.pi*re**2*mec2*NeW
chk=[]
for T in [0.05,0.1,0.5,1.0,2.0]:
    row=phys.estar(T); Sb=K0/phys.beta2(T)*bethe_B(T,I_w,row[5]); chk.append((T,round(row[0],4),round(Sb,4)))
print('Bethe check (T, ESTAR Scol, Bethe):',chk)
Sa=np.zeros(nT)
Sa[ok]=Scol[ok]*(NeAir/NeW)*bethe_B(Tcen[ok],I_a,0.0)/bethe_B(Tcen[ok],I_w,dens[ok])
tail_a=np.zeros(nT); tail_a[ok]=phys.tail_fast(Tcen[ok],DELTA,NeAir)
La=Sa-tail_a
print('Lw(1MeV)',Lw[np.argmin(abs(Tcen-1.0))], 'La(1MeV)',La[np.argmin(abs(Tcen-1.0))], 'Scol ESTAR 1MeV', Scol[np.argmin(abs(Tcen-1.0))])
# Moller sigma-bar matrix for delta production (bins j>=i10, T bins i>=i10)
idx=np.arange(i10,nT)
gl_x,gl_w=np.polynomial.legendre.leggauss(6)
sig=np.zeros((nT,nT))
for j in idx:
    a=Tedges[j]; b=Tedges[j+1]
    eps=0.5*(b-a)*gl_x+0.5*(a+b); wq=0.5*(b-a)*gl_w
    for i in idx:
        Ti=Tcen[i]
        if a>=Ti/2: continue
        dse=phys.moller_dsde(Ti,eps,1)
        sig[i,j]=np.sum(wq*np.where(eps<Ti/2,dse,0.0))
def chain(Qg):
    """Qg: photon-set source per bin (cm^-3). Returns (F_gen0, F_delta) fluence per bin (per Phi0) for T>=10 keV"""
    F0=np.zeros(nT)
    A=np.zeros(nT)
    cs=np.cumsum(Qg[::-1])[::-1]  # sum_{j>=i}
    A[:]=cs-Qg/2   # sum_{j>i} + Q_i/2
    S=Lw_tot
    F0[idx]=(A[idx]/S[idx])*DT
    Fd=np.zeros(nT)
    Fprev=F0.copy()
    for it in range(60):
        Qd=NeW*(Fprev[:,None]*sig).sum(axis=0)   # produced per eps bin j
        cs=np.cumsum(Qd[::-1])[::-1]
        Anew=cs-Qd/2
        Fnew=np.zeros(nT); Fnew[idx]=(Anew[idx]/S[idx])*DT
        Fd+=Fnew
        if Fnew.sum()<1e-7*max(Fd.sum(),1e-30): break
        Fprev=Fnew
    return F0, Fd
def sources(flag_sel):
    Q=np.zeros(nT)
    # primary: exact lines
    if flag_sel in ('p','all'):
        for E,R in zip(ELINES,R_line):
            Q+= R*mu_inc(E)*kn_hist(E)
            # photoelectric
            ib=int(E/DT)
            if E>=0.01: Q[ib]+= R*mu_pe(E)
    if flag_sel in ('s','all'):
        Rw=Rs_fine
        Q+= (Rw*mu_inc(EC))@Pmat
        for j in range(M.NEB):
            if Rw[j]==0: continue
            E=EC[j]
            if E>=0.01:
                Q[int(E/DT)]+= Rw[j]*mu_pe(E)
    return Q
'''
open('/tmp/ana4.py','w').write(ana4)
exec(ana4)
print('Bethe check',chk)
