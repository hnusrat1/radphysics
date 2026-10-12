import numpy as np

L3=np.load('L3.npz')
h_all=L3['h_all']/float(L3['wtot'])   # fractions
wE=float(L3['wE']); wtot=float(L3['wtot'])
Emean_all=wE/wtot
Kcol_Phi=5.352357476400485e-12

edges_e=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
nb_e=25
Egrid=np.arange(0.010,1.4005,0.002); nE=len(Egrid); dE=Egrid[1]-Egrid[0]

def p_of_E(E):
    i=int(np.searchsorted(edges_e,E,side='right')-1); i=max(0,min(nb_e-1,i))
    w=edges_e[i+1]-edges_e[i]
    return h_all[i]/w
p=np.array([p_of_E(E) for E in Egrid])
print("integral p:", np.trapezoid(p,Egrid))

# stopping powers
Ew=np.array([0.01,0.015,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.5])
Sw=np.array([22.36,16.10,12.98,9.00,5.93,4.72,4.10,3.35,3.05,2.63,2.39,2.23,2.12,2.04,1.98,1.93,1.89,1.86,1.84,1.83,1.84],float)
Sa=np.array([20.79,15.00,12.01,8.60,5.46,4.33,3.74,3.05,2.74,2.33,2.11,1.97,1.87,1.80,1.75,1.70,1.64,1.62,1.60,1.57,1.56],float)
def Swater(E): return float(np.exp(np.interp(np.log(E),np.log(Ew),np.log(Sw))))
def Sair(E): return float(np.exp(np.interp(np.log(E),np.log(Ew),np.log(Sa))))
S_col=np.array([Swater(E) for E in Egrid])
S_air=np.array([Sair(E) for E in Egrid])

K_vol_phi=Kcol_Phi*6.242e9
q=K_vol_phi*p/Emean_all

# scale so that sum q*E*dE = K_vol_phi
q_scale=K_vol_phi/np.trapezoid(q*Egrid,Egrid)
q=q*q_scale
print("check sum q E dE:", np.trapezoid(q*Egrid,Egrid))

I=np.zeros(nE); acc=0.0
for i in range(nE-1,-1,-1):
    acc+=q[i]*dE; I[i]=acc
Phi_p=I/S_col
print("primary electron fluence:", np.trapezoid(Phi_p,Egrid))

r0=2.8179403262e-13; mc2=0.5109989461; twopi_r02=2*np.pi*r0*r0
def moller(E,T):
    if T<=0 or T>=E/2: return 0.0
    gamma=1.0+E/mc2; beta2=1.0-1.0/(gamma*gamma)
    A=twopi_r02*mc2/beta2
    term=1.0/(T*T)+1.0/((E-T)*(E-T)) - (2*gamma-1)/(gamma*gamma*T*(E-T)) + ((gamma-1)**2)/(gamma*gamma*mc2*mc2)
    return A*term

def delta_spectrum(Phi_in):
    qd=np.zeros(nE)
    for i in range(nE):
        Eprim=Egrid[i]; Ph=Phi_in[i]
        if Eprim<0.02 or Ph<=0: continue
        Tmax=Eprim/2.0
        # T grid indices j with Egrid[j] <= Tmax
        jmax=int(np.searchsorted(Egrid,Tmax,side='right'))-1
        if jmax<0: continue
        T=Egrid[:jmax+1]
        ds=np.array([moller(Eprim,t) for t in T])
        qd[:jmax+1]+=Ph*3.343e23*ds*dE
    Id=np.zeros(nE); acc=0.0
    for i in range(nE-1,-1,-1):
        acc+=qd[i]*dE; Id[i]=acc
    return Id/S_col

Phi_d1=delta_spectrum(Phi_p)
print("delta1 fluence:", np.trapezoid(Phi_d1,Egrid))
Phi_d2=delta_spectrum(Phi_d1)
print("delta2 fluence:", np.trapezoid(Phi_d2,Egrid))
Phi_tot=Phi_p+Phi_d1+Phi_d2

flux_bin=np.zeros(nb_e)
for i in range(nb_e):
    lo,hi=edges_e[i],edges_e[i+1]
    m=(Egrid>=lo)&(Egrid<hi)
    flux_bin[i]=np.trapezoid(Phi_tot[m],Egrid[m]) if m.sum()>1 else 0.0
print("\nL4_electron_fluence_spectrum:", flux_bin)
print("L4 total:", flux_bin.sum())
print("delta frac:", np.trapezoid(Phi_d1+Phi_d2,Egrid)/np.trapezoid(Phi_tot,Egrid))
print("mean E:", np.trapezoid(Egrid*Phi_tot,Egrid)/np.trapezoid(Phi_tot,Egrid))
Dcheck=1.602e-10*np.trapezoid(Phi_tot*S_col,Egrid)
print("D check:",Dcheck,"Kcol:",Kcol_Phi)

Srad=0.006*Egrid**1.7
print("rad/coll:", np.trapezoid(Phi_tot*Srad,Egrid)/np.trapezoid(Phi_tot*S_col,Egrid))

Delta=0.01; m=Egrid>=Delta
num=np.trapezoid(Phi_tot[m]*S_col[m],Egrid[m]); den=np.trapezoid(Phi_tot[m]*S_air[m],Egrid[m])
PhiD=np.interp(Delta,Egrid,Phi_tot)
saw=(num+PhiD*Delta)/(den+PhiD*Delta)
print("s_w_air:",saw)

def g_rad(E0):
    eg=Egrid[Egrid<=E0]
    if len(eg)<2: return 0.0
    val=np.trapezoid(0.006*eg**1.7/max(Swater(eg[-1]),1e-6),eg) if False else np.trapezoid(np.array([(0.006*e**1.7)/Swater(e) for e in eg]),eg)
    return val/E0
g_num=np.sum(q*Egrid*g_rad(Egrid)*dE); g_den=np.sum(q*Egrid*dE)
g=g_num/g_den
print("g:",g)

# angular model
def mean_cos(E): return min(0.75,0.15+0.5*E/1.2)
def ang_frac(c):
    def meanof(a):
        if abs(a)<1e-9: return 0.0
        return 1.0/np.tanh(a)-1.0/a
    if c<=0.001: return np.full(10,0.1)
    if meanof(8.0)<c: a=8.0
    else:
        lo,hi=0.0,8.0
        for _ in range(60):
            mid=(lo+hi)/2
            if meanof(mid)<c: lo=mid
            else: hi=mid
        a=lo
    edges=np.linspace(-1,1,11)
    vals=np.array([(np.exp(a*edges[k+1])-np.exp(a*edges[k]))/(a) for k in range(10)])
    return vals/vals.sum()
ang=np.zeros(10); totw=np.trapezoid(Phi_tot,Egrid)
for i in range(nE):
    ang+=ang_frac(min(0.78,max(0.0,mean_cos(Egrid[i]))))*Phi_tot[i]*dE/totw
ang/=ang.sum()
print("angular:",ang,"sum:",ang.sum())

np.savez('L4.npz',Egrid=Egrid,Phi_tot=Phi_tot,Phi_p=Phi_p,Phi_d1=Phi_d1,Phi_d2=Phi_d2,flux_bin=flux_bin,ang=ang,g=g,saw=saw,ratio_rc=np.trapezoid(Phi_tot*Srad,Egrid)/np.trapezoid(Phi_tot*S_col,Egrid))
