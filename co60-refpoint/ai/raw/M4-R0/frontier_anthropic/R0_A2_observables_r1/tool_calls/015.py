
ana = r'''
import numpy as np, sys, glob
sys.path.insert(0,'/tmp/work')
import lib
from lib import *
trapz=np.trapezoid
DEPTHS=[0.25,0.5,0.75,1.0,2.0,5.0]
Ebin=0.005; nE=280; nC=10
Ecen=(np.arange(nE)+0.5)*Ebin
Cedges=np.linspace(-1,1,11); Ccen=0.5*(Cedges[1:]+Cedges[:-1])
ST=S  # ESTAR table
def Scol_w(T): return np.exp(np.interp(np.log(T),np.log(ST[:,0]),np.log(ST[:,1])))
def Srad_w(T): return np.exp(np.interp(np.log(T),np.log(ST[:,0]),np.log(ST[:,2])))
def Stot_w(T): return np.exp(np.interp(np.log(T),np.log(ST[:,0]),np.log(ST[:,3])))
def Yield(T): return np.interp(np.log(T),np.log(ST[:,0]),ST[:,5])
def load_H(pattern='/tmp/work/H_r*.npz'):
    files=sorted(glob.glob(pattern)); Htot=None; Ntot=0
    for f in files:
        d=np.load(f)
        if Htot is None: Htot=np.zeros_like(d['H']); dep=d['depths']
        Htot+=d['H']; Ntot+=float(d['N'])
    return Htot, Ntot, dep
def Om_fac(): return lib.Om_p*105.0**2
def arrivals(Hj,Ntot):
    # per-Phi0 scattered fluence in (E-bin, cos-bin) cells
    return Hj.reshape(nE,nC)/Ntot*Om_fac()
def KNw(E):
    # weights for mu grid, E scalar
    mug=np.linspace(-1,1,401)
    f=kn_f(E,mug); w=f/np.sum(f)
    return mug,w
def compton_cell_table(E,c):
    """returns T, cos_e, prob for Compton electrons from photon of energy E arriving with cos c (vectorized over mu grid)"""
    mug,w=KNw(E)
    Ep=E/(1+E/mec2*(1-mug))
    T=E-Ep
    pe=np.sqrt(E*E+Ep*Ep-2*E*Ep*mug)
    cphi=(E-Ep*mug)/pe
    sphi=np.sqrt(np.clip(1-cphi**2,0,1))
    az=np.arange(8)*2*np.pi/8
    ce=cphi[:,None]*c+sphi[:,None]*np.sqrt(max(1-c*c,0))*np.cos(az)[None,:]
    return T, ce, w[:,None]/8*np.ones((1,8))
def electron_sources(Wcells, Wprim):
    """Wcells: (nE,nC) per Phi0 scattered fluence; Wprim: list of (E,W) primary lines.
    Returns dict of histograms on fine T grid (0.002) and direction bins, plus separate primary."""
    dT=0.002; Tg=np.arange(700)*dT+dT/2
    out={}
    def add(key, Tarr, ce_arr, weight):
        # Tarr shape (m,), ce shape (m,8), weight shape (m,8)
        T=np.broadcast_to(Tarr[:,None],ce_arr.shape).ravel()
        cc=ce_arr.ravel(); ww=weight.ravel()
        hT,_=np.histogram(T,bins=np.arange(701)*dT,weights=ww)
        hc,_=np.histogram(np.clip(cc,-1,1),bins=Cedges,weights=ww)
        if key not in out: out[key]=[np.zeros(700),np.zeros(10),0.0,0.0,0.0,0.0]
        out[key][0]+=hT; out[key][1]+=hc
    def add_simple(key,Tarr,cval,weight):
        # weight per T element (density per MeV handled by caller via histogram)
        hT,_=np.histogram(Tarr,bins=np.arange(701)*dT,weights=weight)
        hc,_=np.histogram(np.full(len(Tarr),cval),bins=Cedges,weights=weight)
        if key not in out: out[key]=[np.zeros(700),np.zeros(10),0.0,0.0,0.0,0.0]
        out[key][0]+=hT; out[key][1]+=hc
    # --- primary
    for E,W in Wprim:
        rC=W*mu_inc(np.array([E]))[0]; rP=W*mu_pair(np.array([E]))[0]; rPE=W*mu_pe(np.array([E]))[0]
        T,ce,ww=compton_cell_table(E,1.0)
        add('prim',T,ce,ww*rC)
        if E>1.022:
            Tk=E-1.022
            Tarr=np.linspace(0,Tk,401)[1:]
            wd=np.full(len(Tarr),2*rP/len(Tarr))  # two electrons uniform
            add_simple('prim',Tarr,1.0,wd)
        add_simple('prim',np.array([E-0.0005]),1.0,np.array([rPE]))
    # --- scattered arrivals
    for i in range(nE):
        E=Ecen[i]
        for j in range(nC):
            W=Wcells[i,j]
            if W<=0: continue
            c=Ccen[j]
            rC=W*mu_inc(np.array([E]))[0]; rP=W*mu_pair(np.array([E]))[0]; rPE=W*mu_pe(np.array([E]))[0]
            T,ce,ww=compton_cell_table(E,c)
            add('scat',T,ce,ww*rC)
            if E>1.022:
                Tk=E-1.022
                Tarr=np.linspace(0,Tk,41)[1:]
                add_simple('scat',Tarr,c,np.full(len(Tarr),2*rP/len(Tarr)))
            add_simple('scat',np.array([E-0.0005]),c,np.array([rPE]))
    return out, Tg, dT
'''
open('/tmp/work/ana.py','w').write(ana)
print("ok")
