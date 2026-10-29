import numpy as np
from scipy.interpolate import interp1d
me=0.51099895
r0=2.8179403e-13; NA=6.02214076e23; Ne_w=NA*(10.0/18.015268)
def trapz(y,x): return np.trapezoid(y,x)
d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
_Em=d[:,0]; _incoh=d[:,2]; _pe=d[:,3]; _pair=d[:,4]+d[:,5]
f_incoh=interp1d(_Em,_incoh,bounds_error=False,fill_value=(_incoh[0],_incoh[-1]))
f_pe=interp1d(_Em,_pe,bounds_error=False,fill_value=(_pe[0],_pe[-1]))
f_pair=interp1d(_Em,_pair,bounds_error=False,fill_value=(_pair[0],_pair[-1]))
d2=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
f_Sc=interp1d(d2[:,0],d2[:,1],bounds_error=False,fill_value=(d2[0,1],d2[-1,1]))
f_St=interp1d(d2[:,0],d2[:,3],bounds_error=False,fill_value=(d2[0,3],d2[-1,3]))
PH_EDGES=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
                   0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
phc=(PH_EDGES[:-1]+PH_EDGES[1:])/2
E1=1.1732; E2=1.3325
def kn_ratio(E,mu):
    a=E/me; r=1/(1+a*(1-mu)); return r*r*(r+1/r-(1-mu**2))
def compton_pdf_grid(E, EG):
    a=E/me
    mu=np.linspace(-1+1e-9,1-1e-9,6000)
    r=1/(1+a*(1-mu)); Ep=E*r; Ee=E-Ep
    dmu=mu[1]-mu[0]; w=2*np.pi*kn_ratio(E,mu)*dmu; tot=w.sum()
    edges=np.append(EG,EG[-1]+0.05)
    pdf_hist,_=np.histogram(Ee,bins=edges,weights=w/tot)
    return pdf_hist/np.diff(edges)
def moller_dsdT(Ep,T):
    gamma=(Ep+me)/me; b2=1-1/gamma**2
    pref=2*np.pi*r0**2*me/b2
    return pref*(1/T**2+1/(Ep-T)**2-(2*gamma-1)/gamma**2/(T*(Ep-T))+((gamma-1)/gamma)**2/me**2)
def solve_fluence(S0, EG, iters=60):
    N=len(EG); dE=np.gradient(EG); St=f_St(EG)
    def cumint(f):
        vals=f*dE; c=np.zeros(N); c[-1]=vals[-1]/2
        for i in range(N-2,-1,-1): c[i]=c[i+1]+(vals[i]+vals[i+1])/2
        return c
    K=np.zeros((N,N))
    for j in range(N):
        iend=np.searchsorted(EG,EG[j]/2,side='right')
        if iend>0: K[j,:iend]=Ne_w*moller_dsdT(EG[j],EG[:iend])
    I0=cumint(S0); Phi=np.zeros(N)
    for it in range(iters):
        Sd=K.T.dot(Phi*dE); Phi=np.maximum((I0+cumint(Sd))/St,0)
    return Phi, I0/St

EG=np.exp(np.linspace(np.log(0.010),np.log(1.4),150)); widths=np.gradient(EG)
f1=0.5*np.exp(-0.0653626*5); f2=0.5*np.exp(-0.06122524*5)
groups_prim=[(E1,f1),(E2,f2)]
# build S0 primary-only with and without PE
S0=np.zeros(len(EG))
for E,w in groups_prim:
    S0+=w*f_incoh(E)*compton_pdf_grid(E,EG)
    S0[np.argmin(abs(EG-E))]+=w*f_pe(E)/widths[np.argmin(abs(EG-E))]
Phi,Phi0=solve_fluence(S0,EG)
# print S0 and Phi at selected energies
sel=np.arange(0,150,10)
print("EG:",np.round(EG[sel],3))
print("S0:",np.round(S0[sel],6))
print("Phi:",np.round(Phi[sel],6))
print("Phi0:",np.round(Phi0[sel],6))
# Also compute pdf for E1 alone on grid to see its shape
pdf1=compton_pdf_grid(E1,EG); pdf2=compton_pdf_grid(E2,EG)
print("pdf1:",np.round(pdf1[sel],4))
print("pdf2:",np.round(pdf2[sel],4))
print("integral pdf1:",trapz(pdf1,EG))