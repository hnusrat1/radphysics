import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

ph = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
Eph = ph[:,0]; coh=ph[:,1]; inc=ph[:,2]; pe=ph[:,3]; pn=ph[:,4]; pel=ph[:,5]; tot=ph[:,6]
def mklogf(x,y):
    ok=(y>0)&(x>0)
    return interp1d(np.log(x[ok]),np.log(y[ok]),kind='linear',bounds_error=False,fill_value='extrapolate')
fcoh=mklogf(Eph,coh); finc=mklogf(Eph,inc); fpe=mklogf(Eph,pe); fpn=mklogf(Eph,pn); fpel=mklogf(Eph,pel); ftot=mklogf(Eph,tot)
def mu_coh(E): return float(np.exp(fcoh(np.log(float(E)))))
def mu_inc(E): return float(np.exp(finc(np.log(float(E)))))
def mu_pe(E):  return float(np.exp(fpe(np.log(float(E)))))
def mu_pn(E):  return float(np.exp(fpn(np.log(float(E)))))
def mu_pel(E): return float(np.exp(fpel(np.log(float(E)))))
def mu_tot(E): return float(np.exp(ftot(np.log(float(E)))))

# electron data
el = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=el[:,0]; Sc=el[:,1]; Sr=el[:,2]; St=el[:,3]; R=el[:,4]; Y=el[:,5]
def Scol(E): return float(np.interp(E,Ee,Sc))
def Srad(E): return float(np.interp(E,Ee,Sr))
def Yrad(E): return float(np.interp(E,Ee,Y))

mec2 = 0.5109989461

def KN_weight(cosT, E):
    """unnormalized KN differential in cos(theta_s): (E'/E)^2 (E'/E+E/E'-sin^2)"""
    a = E/mec2
    Ep = E/(1+a*(1-cosT))
    r = Ep/E
    return r*r*(r + 1/r - (1-cosT*cosT))

def KN_pdf_max(E):
    # max at cosT=-1
    return KN_weight(-1.0,E)

def sample_KN(E, n=1, rng=np.random):
    """sample cos(theta_s) from KN via rejection, returns cosT and E' and T"""
    cosT = np.empty(n)
    fmax = KN_pdf_max(E)
    got = np.zeros(n,bool)
    while not got.all():
        c = rng.uniform(-1,1,n)
        accept = (rng.uniform(0,1,n)*fmax) < KN_weight(c,E)
        new = (~got)&accept
        cosT[new]=c[new]; got[new]=True
    a=E/mec2
    Ep = E/(1+a*(1-cosT))
    T = E-Ep
    return cosT, Ep, T

def KN_averages(E):
    """<T> and <T(1-Y(T))> over KN distribution"""
    fmax = KN_pdf_max(E)
    # integrate numerically over cosT
    def integ_T(ct): return KN_weight(ct,E)* (E - E/(1+(E/mec2)*(1-ct)))
    def integ_TY(ct):
        T = E - E/(1+(E/mec2)*(1-ct))
        return KN_weight(ct,E)* T*(1-Yrad(T))
    Z = quad(lambda ct: KN_weight(ct,E), -1, 1)[0]
    mT = quad(integ_T, -1,1)[0]/Z
    mTY= quad(integ_TY,-1,1)[0]/Z
    return mT, mTY

# compute mu_tr*E and mu_en*E as functions of E
def mutrE(E):
    mT,_=KN_averages(E)
    val = mu_pe(E)*(E-0.0005) + mu_inc(E)*mT
    if E>1.022:
        val += (mu_pn(E)+mu_pel(E))*(E-1.022)
    return val
def muenE(E):
    mT,mTY = KN_averages(E)
    val = mu_pe(E)*(E-0.0005)*(1-Yrad(E-0.0005)) + mu_inc(E)*mTY
    if E>1.022:
        Te=(E-1.022)/2.0
        val += (mu_pn(E)+mu_pel(E))*2*Te*(1-Yrad(Te))
    return val

for E in [1.1732,1.3325]:
    mT,mTY=KN_averages(E)
    print(f"E={E}: <T>={mT:.4f} <T(1-Y)>={mTY:.4f} mu_tr*E={mutrE(E):.5f} mu_en*E={muenE(E):.5f} g_eff={1-muenE(E)/mutrE(E):.5f}")
    print("  fractions of mutrE: pe %.5f inc %.5f pair %.5f"%(mu_pe(E)*(E-0.0005)/mutrE(E), mu_inc(E)*mT/mutrE(E), (mu_pn(E)+mu_pel(E))*(E-1.022)/mutrE(E) if E>1.022 else 0))
# check mu_tot vs sum
print("mu_tot check:", mu_tot(1.1732), mu_coh(1.1732)+mu_inc(1.1732)+mu_pe(1.1732)+mu_pn(1.1732)+mu_pel(1.1732))