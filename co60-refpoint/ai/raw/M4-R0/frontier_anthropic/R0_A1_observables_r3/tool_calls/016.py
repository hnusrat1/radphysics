
full = r'''
import sys, pickle, json, numpy as np
sys.path.insert(0,'/tmp/work')
from physics import *
import physics as P
d_=pickle.load(open('/tmp/work/tot.pkl','rb'))
tot=d_['tot']; Ntot=d_['N']; zpts=d_['zpts']
im=zpts.index(5.0)
SC=P.SF/Ntot
sg=np.load('/tmp/work/main5_seg.npz'); Es=sg['E']; ds=sg['d']; Ws=sg['W']*SC
rng=np.random.default_rng(99)
prim_w=0.5*np.exp(-np.array([P.mutot(E) for E in lines])*5.0)
out={}
def electrons(E,dv,W,labels=False):
    Ts=[];cs=[];ws=[];lab=[]
    wc=W*muC(E); eps,cc=sample_KN(E,rng); phi=2*np.pi*rng.random(E.size)
    dout=rotate(dv,cc,phi); Eout=E*eps; T=E-Eout
    pe=E[:,None]*dv-Eout[:,None]*dout; nrm=np.linalg.norm(pe,axis=1)
    Ts.append(T); cs.append(pe[:,2]/nrm); ws.append(wc); lab.append(np.zeros(E.size,int))
    wp=W*mupe(E); Ts.append(E.copy()); cs.append(dv[:,2]); ws.append(wp); lab.append(np.ones(E.size,int))
    wpr=W*mupair(E); avail=np.clip(E-2*me,0,None); u=rng.random(E.size)
    Ts.append(u*avail); cs.append(dv[:,2]); ws.append(wpr); lab.append(np.full(E.size,2))   # e-
    Ts.append((1-u)*avail); cs.append(dv[:,2]); ws.append(wpr); lab.append(np.full(E.size,3))  # e+
    return np.concatenate(Ts),np.concatenate(cs),np.concatenate(ws),np.concatenate(lab)
def spec(T,w,edges):
    h,_=np.histogram(T,bins=edges,weights=w); return h/h.sum()
# ---- scattered electron sources
Ta,ca,wa,la=electrons(Es,ds,Ws)
# ---- primary electron sources (many samples)
nrep=400000
Ep_=np.repeat(lines.astype(float),nrep); dp_=np.tile([0,0,1.0],(2*nrep,1)); Wp_=np.repeat(prim_w/nrep,nrep)
Tp_,cp_,wp_,lp_=electrons(Ep_,dp_,Wp_)
Tall=np.concatenate([Ta,Tp_]); call=np.concatenate([ca,cp_]); wall=np.concatenate([wa,wp_]); lall=np.concatenate([la,lp_])
m_all=Tall>=0.0
out['L3_initial_spectrum']=spec(Tall,wall,ebE).tolist()
out['L3_initial_spectrum_parent_primary']=spec(Tp_,wp_,ebE).tolist()
out['L3_mean_initial_energy_MeV']=float(np.sum(Tall*wall)/np.sum(wall))
out['L3_mean_initial_energy_parent_primary_MeV']=float(np.sum(Tp_*wp_)/np.sum(wp_))
hc,_=np.histogram(call,bins=cb,weights=wall); out['L3_initial_direction']=(hc/hc.sum()).tolist()
# ---- L4 source: e- only (Compton, photo, pair e-) from both primary and scattered
mE=(lall!=3)
h=0.002
Tk=0.01+h*np.arange(696)   # 0.01 .. 1.40
n=len(Tk)
edges=np.concatenate([Tk-h/2,[Tk[-1]+h/2]])
Qg_hist,_=np.histogram(Tall[mE],bins=edges,weights=wall[mE])
Qg=Qg_hist/h
# angular weights for L4: L(T0)= int_{0.01}^{T0} dE/S
Sw=Scol_bethe(Tk,Iw,neg_w); Srad=rad_S(Tk)
Lcum=np.concatenate([[0],np.cumsum(h/Sw)])   # track length from 0.01 to Tk[k]
Lk=np.interp(Tall[mE],np.concatenate([[0.01],Tk]),np.concatenate([[0],Lcum[:-0 or None]]) if False else np.concatenate([[0],Lcum]) )
# ---- slowing down with delta rays
Psi=np.zeros(n); Qd=np.zeros(n); Qtot=np.zeros(n)
cum=0.0
for k in range(n-1,-1,-1):
    T=Tk[k]
    js=np.arange(k+1,n); js=js[Tk[js]>2*T]
    if js.size>0:
        Tp=Tk[js]; tau=Tp/me; b2=tau*(tau+2)/(tau+1)**2
        Nn=2*np.pi*re**2*me*ne_w/(b2*T*T)*np.clip(1-b2*T/(Tp/2),0,None)
        Qd[k]=np.sum(Psi[js]*Nn*h)
    Qtot[k]=Qg[k]+Qd[k]
    Psi[k]=(cum+Qtot[k]*h/2)/Sw[k]
    cum+=Qtot[k]*h
# gamma-only (primary electron tracks)
Psig=np.zeros(n); cum=0.0
for k in range(n-1,-1,-1):
    Psig[k]=(cum+Qg[k]*h/2)/Sw[k]; cum+=Qg[k]*h
Psid=Psi-Psig
L4_fluence=np.sum(Psi*h)
out['L4_electron_fluence_per_phi0']=float(L4_fluence)
bins_fl,_=np.histogram(Tk,bins=ebE,weights=Psi*h)
out['L4_electron_fluence_spectrum']=bins_fl.tolist()
out['L4_delta_fraction']=float(np.sum(Psid*h)/L4_fluence)
out['L4_mean_electron_energy_MeV']=float(np.sum(Tk*Psi*h)/L4_fluence)
out['L4_radiative_to_collision']=float(np.sum(Psi*Srad*h)/np.sum(Psi*Sw*h))
# angular for L4 (primary tracks weighted by CSDA track length, deltas assumed co-directional)
hl,_=np.histogram(call[mE],bins=cb,weights=wall[mE]*Lk)
out['L4_angular']=(hl/hl.sum()).tolist()
# ---- Layer 5 numbers
ET_p=np.sum(prim_w*ET(lines)); KC_p=np.sum(prim_w*KC(lines))
Ks=SC*tot['A_K'][im]; Kcs=SC*tot['A_Kc'][im]
K_tot=ET_p+Ks; Kc_tot=KC_p+Kcs
g=1-Kc_tot/K_tot
conv=1.602177e-10
out['L5_K_over_phi0_Gy_cm2']=float(K_tot*conv)
out['L5_Kcol_over_phi0_Gy_cm2']=float(Kc_tot*conv)
out['L5_g']=float(g)
out['L5_dose_fraction_parent_scattered']=float(SC*tot['A_Kc_scat'][im]/Kc_tot)
# SA water/air at 5cm
Dl=0.01
def Lrestr(Tv,neg,I):
    S=Scol_bethe(Tv,I,neg)
    Kc_=2*np.pi*re**2*me*neg
    tau=Tv/me; b2=tau*(tau+2)/(tau+1)**2
    Tmax=Tv/2
    corr=(Kc_/b2)*(np.log(np.maximum(Tmax/Dl,1))-b2*(1-Dl/Tmax))
    L=np.where(Tmax>Dl, S-corr, S)
    return L
Lw=Lrestr(Tk,neg_w,Iw); La=Lrestr(Tk,neg_a,Ia)
Sw_d=Scol_bethe(np.array([Dl]),Iw,neg_w)[0]; Sa_d=Scol_bethe(np.array([Dl]),Ia,neg_a)[0]
Psi_d=Psi[0]
num=np.sum(Psi*Lw*h)+Psi_d*Dl*Sw_d
den=np.sum(Psi*La*h)+Psi_d*Dl*Sa_d
out['L5_s_w_air']=float(num/den)
# ---- depth-dose via forward kernel (Kcol(z) from MC)
Kz=np.array([np.sum(prim_w*0)+0 for _ in zpts])
def Kcol_prim(z): return 0.5*np.sum(np.exp(-np.array([P.mutot(E) for E in lines])*z)*KC(lines))
Kcol_z=np.array([Kcol_prim(z)+SC*tot['A_Kc'][j] for j,z in enumerate(zpts)])
out['_Kcol_z']=dict(zip([float(z) for z in zpts],[float(x) for x in Kcol_z]))
zt=np.array(zpts)
def Kint(z):
    return np.interp(z,zt,Kcol_z)   # zt starts at 0.05 (constant below)
def Dconv(m, zg):
    Dz=np.zeros(len(zg))
    zp=np.linspace(0,zg[-1],3001); dz=zp[1]-zp[0]
    Kp=Kint(zp); Kp[zp<zt[0]]=Kcol_z[0]
    for i,z in enumerate(zg):
        mask=zp<=z
        x=z-zp[mask]
        Dz[i]=np.sum(Kp[mask]*np.exp(-x/m)/m*dz)
    return Dz
zg=np.linspace(0.05,4.0,396)
best=None
for m in [0.15,0.2,0.25,0.3]:
    Dz=Dconv(m,zg)
    imax=np.argmax(Dz)
    D5=Dconv(m,np.array([5.0]))[0]
    pdd5=100*D5/Dz[imax]
    print("m",m,"dmax",round(zg[imax],3),"D(dmax)/K(dmax)",round(Dz[imax]/Kint(zg[imax]),4),"beta5",round(D5/Kint(5.0),4),"PDD5",round(pdd5,2), "D(0.05)/Dmax", round(Dz[0]/Dz[imax],3))
out['_scan']=None
'''
open('/tmp/work/full.py','w').write(full)
import subprocess
r=subprocess.run(['python3','/tmp/work/full.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-3000:], r.stderr[-3000:])
