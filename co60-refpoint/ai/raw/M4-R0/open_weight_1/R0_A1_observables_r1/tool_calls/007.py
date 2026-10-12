import numpy as np

L3=np.load('L3.npz')
h_all=L3['h_all']/float(L3['wtot'])
wE=float(L3['wE']); wtot=float(L3['wtot']); Emean_all=wE/wtot
Kcol_Phi=5.352357476400485e-12
edges_e=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
nb_e=25
Egrid=np.arange(0.010,1.4005,0.002); nE=len(Egrid); dE=Egrid[1]-Egrid[0]
def p_of_E(E):
    i=int(np.searchsorted(edges_e,E,side='right')-1); i=max(0,min(nb_e-1,i))
    return h_all[i]/(edges_e[i+1]-edges_e[i])
p=np.array([p_of_E(E) for E in Egrid])
K_vol_phi=Kcol_Phi*6.242e9
q=K_vol_phi*p/Emean_all
q=q*K_vol_phi/np.trapezoid(q*Egrid,Egrid)

Ew=np.array([0.01,0.015,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.5])
Sw=np.array([22.36,16.10,12.98,9.00,5.93,4.72,4.10,3.35,3.05,2.63,2.39,2.23,2.12,2.04,1.98,1.93,1.89,1.86,1.84,1.83,1.84],float)
def Swater(E): return float(np.exp(np.interp(np.log(E),np.log(Ew),np.log(Sw))))
S_col=np.array([Swater(E) for E in Egrid])

r0=2.8179403262e-13; mc2=0.5109989461; twopi_r02=2*np.pi*r0*r0
def moller(E,T):
    if T<=0 or T>=E/2: return 0.0
    gamma=1.0+E/mc2; beta2=1.0-1.0/(gamma*gamma)
    A=twopi_r02*mc2/beta2
    term=1.0/(T*T)+1.0/((E-T)*(E-T)) - (2*gamma-1)/(gamma*gamma*T*(E-T)) + ((gamma-1)**2)/(gamma*gamma*mc2*mc2)
    return A*term

# restricted stopping power L_Delta(E): S_col - n_e * int_Delta^{E/2} T dsigma/dT dT
Delta=0.01
Lres=np.zeros(nE)
for i in range(nE):
    E=Egrid[i]
    if E/2<=Delta: Lres[i]=S_col[i]
    else:
        Te=Egrid[(Egrid>=Delta)&(Egrid<=E/2)]
        if len(Te)>0:
            vals=np.array([t*3.343e23*moller(E,t) for t in Te])
            Lres[i]=S_col[i]-np.trapezoid(vals,Te)
        else:
            Lres[i]=S_col[i]

def cumint_hi(arr, dx):
    out=np.zeros(len(arr)); acc=0.0
    for i in range(len(arr)-1,-1,-1):
        acc+=arr[i]*dx; out[i]=acc
    return out

# iteration
Iph=cumint_hi(q,dE)
Phi=np.zeros(nE)
Phi-=Phi
# start
it=0
for it in range(6):
    qk=np.zeros(nE)
    # qk(Tj)=n_e sum_i Phi_i moller(Ei,Tj) dE for Ei>=2Tj
    for j in range(nE):
        T=Egrid[j]
        imin=int(np.searchsorted(Egrid,2*T,side='left'))
        if imin>=nE: break
        ds=np.array([moller(Egrid[i],T) for i in range(imin,nE)])
        qk[j]=3.343e23*np.sum(Phi[imin:nE]*ds)*dE
    Ik=cumint_hi(qk,dE)
    Phi_new=(Iph+Ik)/Lres
    diff=np.max(np.abs(Phi_new-Phi))/max(1e-12,np.max(np.abs(Phi_new)))
    Phi=Phi_new
    print("iteration",it+1,"diff",diff)
    if diff<1e-4: break

# no-delta primary spectrum (with Lres)
Phi_p=(Iph)/Lres
Phi_delta=Phi-Phi_p
print("Phi total fluence:",np.trapezoid(Phi,Egrid))
print("Phi p fluence:",np.trapezoid(Phi_p,Egrid))
print("delta fluence:",np.trapezoid(Phi_delta,Egrid))
print("delta fraction:",np.trapezoid(Phi_delta,Egrid)/np.trapezoid(Phi,Egrid))
# dose check: dose = int Phi Lres dE + track end (Phi(Delta)*Delta)
D1=np.trapezoid(Phi*Lres,Egrid)
PhiD=np.interp(Delta,Egrid,Phi)
D_trackend=PhiD*Delta
print("D from Phi*Lres + trackend:",1.602e-10*(D1+D_trackend),"Kcol:",Kcol_Phi)
meanE=np.trapezoid(Egrid*Phi,Egrid)/np.trapezoid(Phi,Egrid)
print("mean E all electrons:",meanE)
# radiative to collision
Srad=0.006*Egrid**1.7
print("rad/coll:",np.trapezoid(Phi*Srad,Egrid)/np.trapezoid(Phi*S_col,Egrid))
# bin
flux_bin=np.zeros(nb_e)
for i in range(nb_e):
    lo,hi=edges_e[i],edges_e[i+1]
    m=(Egrid>=lo)&(Egrid<hi)
    flux_bin[i]=np.trapezoid(Phi[m],Egrid[m]) if m.sum()>1 else 0.0
np.savez('L4b.npz',Egrid=Egrid,Phi=Phi,Phi_p=Phi_p,Phi_delta=Phi_delta,flux_bin=flux_bin,Lres=Lres)
print("binned spectrum:",flux_bin)
print("total:",flux_bin.sum())
