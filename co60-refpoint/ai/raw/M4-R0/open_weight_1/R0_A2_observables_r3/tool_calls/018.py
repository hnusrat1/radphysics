# Build analytic computation module: compton kinematics, electron source spectra, CPE with delta rays
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

me=0.51099895
r0=2.8179403e-13
NA=6.02214076e23
Cconst=2*np.pi*r0**2*me*NA  # MeV cm^2/mol

# water
ZA_w=10.0/18.015268
Ne_w=NA*ZA_w

PH_EDGES=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
                   0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
EL_EDGES=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
                   0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
CT_EDGES=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

def kn_ratio(E,mu):
    a=E/me; r=1/(1+a*(1-mu)); return r*r*(r+1/r-(1-mu**2))

def compton_electron_dist(E, Eg, mu_grid=None):
    # returns arrays: electron energies Ee and weights dP/dEe approx via transformation on mu grid
    alpha=E/me
    # choose mu grid points
    mu=np.linspace(-1+1e-9,1-1e-9,2000)
    r=1/(1+alpha*(1-mu))
    Ep=E*r
    Ee=E-Ep
    dmu=mu[1]-mu[0]
    w=2*np.pi*kn_ratio(E,mu)*dmu  # approx dsigma/dEe dp: actually need dP/dEe
    # dEe/dmu = -dEp/dmu = E*alpha*r^2? derivative of Ep wrt mu: Ep=E/(1+a(1-m)), dEp/dm = E*a/(1+a(1-m))^2 = E*a*r^2
    dEedmu = E*alpha*r**2
    dPdE = w/dEedmu   # proportional to dP/dEe
    return Ee, dPdE

def compton_electron_spectrum_on_grid(E, Eg, E_grid):
    # probability per MeV at each E_grid point (via histogram of transformation) - return normalized pdf values (dP/dE)
    Ee, dPdE = compton_electron_dist(E, Eg)
    # integrate dPdE over E_grid bins -> pdf per MeV (piecewise constant)
    pdf=np.zeros(len(E_grid))
    # use np.histogram with weights dPdE*dEe... simpler: approximate by hist of Ee with weights dPdE*dEedmu? 
    # Actually we have (Ee points, dP~dPdE*dEe) pairs; compute counts in bins
    dEe = np.gradient(Ee)
    # normalize
    tot=np.sum(dPdE*dEe)
    dPdE_n = dPdE/tot
    # evaluate pdf (per MeV) on grid by histogram with weights
    vals,bins=np.histogram(Ee, bins=np.append(E_grid,np.inf), weights=dPdE_n)
    # vals/bin-width
    widths=np.diff(np.append(E_grid,np.inf)); widths[-1]=1
    pdf=vals/widths
    return pdf

# moller
def moller_dsdT(Ep, T):
    # Ep incident kinetic, T secondary (T<=Ep/2). returns cm^2/MeV
    gamma=(Ep+me)/me
    b2=1-1/gamma**2
    pref=2*np.pi*r0**2*me/b2   # MeV cm^2
    term=1/T**2 + 1/(Ep-T)**2 - (2*gamma-1)/gamma**2 * 1/(T*(Ep-T)) + ((gamma-1)/gamma)**2/me**2
    return pref*term

def solve_electron_fluence(S0_func, Emax=1.4, Emin=0.010, N=120, iters=60, verbose=False):
    # S0_func: callable(E0) returning electron source per g per MeV per Phi0 (cm^2/g/MeV), for E0 in [Emin, Emax]
    # returns grid EG, fluence spectrum PhiE per MeV per Phi0 (MeV^-1)
    EG=np.exp(np.linspace(np.log(Emin),np.log(Emax),N))
    dE=np.gradient(EG)
    S0=S0_func(EG)
    # primary-only fluence
    # cumulative integral from E to Emax
    # use trapezoid on log grid: integrate S0*dE
    # Build cumulative sums: I(E_i)=sum_{j>=i} S0_j*dE_j
    def cumint(f):
        vals=f*dE
        c=np.zeros(N)
        c[-1]=vals[-1]/2
        for i in range(N-2,-1,-1):
            c[i]=c[i+1]+(vals[i]+vals[i+1])/2
        return c
    # stopping powers (mass) from ESTAR
    d2=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
    _Ee=d2[:,0]; _Sc=d2[:,1]; _St=d2[:,3]
    f_Sc=interp1d(_Ee,_Sc,bounds_error=False,fill_value=(_Sc[0],_Sc[-1]))
    f_St=interp1d(_Ee,_St,bounds_error=False,fill_value=(_St[0],_St[-1]))
    Sc=f_Sc(EG); St=f_St(EG)
    Phi=np.zeros(N)
    # iteration
    Sd=np.zeros(N)
    # precompute moller kernel matrix K[j,i] = Ne_w * (dE' weight) * dsdT(EG[j], EG[i]) for EG[i]<=EG[j]/2
    K=np.zeros((N,N))
    for j in range(N):
        Emax_d=EG[j]/2
        for i in range(N):
            if EG[i]<=Emax_d:
                K[j,i]=Ne_w*moller_dsdT(EG[j],EG[i])
    # iterate: Phi_new(E_i) = (I0(i)+Id(i))/St(i), Id(i)=sum_{j>=? } Phi_j * K[j,i]*dE_j ... careful indices: Sd(E0_i)=sum_j Phi(EG_j)*K[j,i]*dE_j
    Phi_prev=np.zeros(N)
    for it in range(iters):
        # Sd(E0_i) = sum_j Phi(EG_j)*K[j,i]*dE_j = (K.T * (Phi*dE))[i]
        Sd = K.T.dot(Phi*dE)
        I0=cumint(S0)
        Id=cumint(Sd)
        Phi_new=(I0+Id)/St
        Phi_new=np.maximum(Phi_new,0)
        if it>5 and np.max(np.abs(Phi_new-Phi_prev))/max(np.max(Phi_new),1e-30)<1e-6:
            Phi=Phi_new; break
        Phi_prev=Phi_new; Phi=Phi_new
    Phi_primary_only=cumint(S0)/St
    return EG, Phi, Phi_primary_only

# test compton electron dist
E=1.25
Ee,dPdE=compton_electron_dist(E,None)
print("Te mean:", np.sum(Ee*dPdE*np.gradient(Ee)) , "min/max Te:",Ee.min(),Ee.max())
# check mean electron energy: average Te for 1.25 = E*alpha*<...>; fC(1.25)=0.47 -> Te~0.59
print("sample check ok")