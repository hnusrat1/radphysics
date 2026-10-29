
Delta=0.01
def Ldelta(E,K,Sc):
    Tmax=E/2
    dd=np.where(Tmax>Delta, K/(1-0)*0+0, 0)
    b2=1-1/(E/mc2_+1)**2
    extra=(K/b2)*(np.log(np.maximum(Tmax,Delta)/Delta)-b2*(1-Delta/np.maximum(Tmax,Delta)))
    extra=np.where(Tmax>Delta, extra, 0.0)
    return Sc-extra
Lw=Ldelta(Ec,Kw,Scw)
La=Ldelta(Ec,Ka,Scol_a)
m=Ec>=Delta
Phi_D=np.interp(Delta,Ec,Phi)
LwD=np.interp(Delta,Ec,Lw); LaD=np.interp(Delta,Ec,La)
Tend_w=Delta*Phi_D*LwD; Tend_a=Delta*Phi_D*LaD
sw=(np.sum((Phi*dE*Lw)[m])+Tend_w)/(np.sum((Phi*dE*La)[m])+Tend_a)
print("s_w,air (SA, Delta=10keV, incl track-end)", sw, " without track-end:", np.sum((Phi*dE*Lw)[m])/np.sum((Phi*dE*La)[m]))
print("Tend share", Tend_w/np.sum((Phi*dE*Lw)[m]))
# check restricted water/air ratio at 0.5 MeV for sanity
print("Lw(0.5),La(0.5)", Ldelta(np.array([0.5]),Kw,Scol(np.array([0.5]),75,ne_w))[0], Ldelta(np.array([0.5]),Ka,Scol(np.array([0.5]),85.7,ne_a))[0])
