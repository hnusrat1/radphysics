
def Srestr(T, Delta, I_eV, ZA, delta=None):
    """Restricted collision stopping power (MeV cm2/g), ICRU37 electron formula (Delta, T in MeV)."""
    T=np.asarray(T,float); Delta=np.minimum(np.asarray(Delta,float), T/2)
    tau=T/mc2; b2=1-1/(1+tau)**2; Ip=I_eV*1e-6/mc2; D=Delta/mc2
    if delta is None: delta=np.zeros_like(T)
    F = -1-b2+np.log((tau-D)*D)+tau/(tau-D)+(D**2/2+(2*tau+1)*np.log(1-D/tau))/(tau+1)**2
    K = 2*np.pi*re**2*mc2*6.02214076e23*ZA
    return K/b2*(np.log(2*(tau+2)/Ip**2)+F-delta)
# check unrestricted (Delta=T/2) vs ESTAR water
for T in [0.1,0.5,1.0,1.25]:
    d = DEL(T)
    print(T, Srestr(T,T/2,75.0,0.55508,d), S_col(T))
