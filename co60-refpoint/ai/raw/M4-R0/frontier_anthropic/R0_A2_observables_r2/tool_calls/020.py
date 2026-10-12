
ana3 = r'''
# ---------- electron grid ----------
DT=0.005
Tedges=np.arange(0,1.4+1e-9,DT)           # 0..1.4 MeV, 280 edges -> 280 bins
Tcen=(Tedges[:-1]+Tedges[1:])/2
nT=len(Tcen)
i10=int(round(0.01/DT))                    # first bin at 10 keV
def kn_hist(E):
    # probability per T-bin for Compton electrons from photon energy E (exact Gauss-Legendre in mu)
    g,w=np.polynomial.legendre.leggauss(3000)
    k=E/mec2
    e=1/(1+k*(1-g))
    dsig=e*e*(e+1/e-(1-g*g))*w   # dsigma/dmu up to const
    T=E*(1-e)
    h,_=np.histogram(T,bins=Tedges,weights=dsig)
    return h/h.sum()
# photon fine bin -> electron source matrix (Compton), using bin centers of photon fine grid
Eedges_fine=np.arange(0,M.NEB+1)*M.DE
Pmat=np.zeros((M.NEB,nT))
for j in range(M.NEB):
    Ej=EC[j]
    if Ej<0.0011: continue
    Pmat[j]=kn_hist(Ej)
# restricted and unrestricted stopping (water) from ESTAR minus Moller tail above Delta
DELTA=0.01
def S_estar(T):
    return phys.estar(T)  # Scol,Srad,Stot,CSDA,Yrad,delta
Scol_c=np.zeros(nT); Srad_c=np.zeros(nT); delta_c=np.zeros(nT)
for i,T in enumerate(Tcen):
    if T<0.0099: 
        continue
    sc,sr,st,cs,yr,dl=S_estar(T)
    Scol_c[i]=sc; Srad_c[i]=sr; delta_c[i]=dl
# tail above DELTA (Moller)  -- restricted water loss
tail_w=np.array([phys.above(T,DELTA,1,NeW) if T>0.0099 else 0 for T in Tcen])
Lw=Scol_c-tail_w
# air: Bethe scaling of ESTAR water with I_air and density-effect ~0 for air
I_w=75e-6; I_a=85.7e-6
def bethe_B(T,I,Zoa,delta):
    tau=T/mec2; b2=phys.beta2(T)
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
    return np.log(tau**2*(tau+2)/(2*(I/mec2)**2))+F-delta
K0=2*np.pi*re**2*mec2*NeW   # MeV cm^2 * cm^-3 (per g/cm3 density 1)
# check Bethe reproduces ESTAR water
chk=[]
for T in [0.05,0.1,0.5,1.0,2.0]:
    sc=S_estar(T)[0]; dl=S_estar(T)[5]
    Sb=K0/phys.beta2(T)*bethe_B(T,I_w,0,dl)
    chk.append((T,round(sc,4),round(Sb,4)))
print('Bethe check (T, ESTAR Scol, Bethe-calc):',chk)
Sa_c=np.zeros(nT)
for i,T in enumerate(Tcen):
    if T<0.0099: continue
    Sa_c[i]=Scol_c[i]*(NeAir/NeW)*bethe_B(T,I_a,0,0.0)/bethe_B(T,I_w,0,delta_c[i])
    # (Bethe scaling: same K0 shape with N_e ratio, ratio of log terms)
tail_a=np.array([phys.above(T,DELTA,1,NeAir) if T>0.0099 else 0 for T in Tcen])
La=Sa_c-tail_a
'''
s=open('/tmp/ana.py').read()+ana3
open('/tmp/ana.py','w').write(s)
exec(s)
print(Lw[[i10,i10+50,i10+150,i10+195]], La[[i10,i10+50,i10+150,i10+195]])
print('ESTAR Scol at 1MeV',Scol_c[200],'Lw',Lw[200],'La',La[200], 'air/water collision', Sa_c[200]/Scol_c[200])
