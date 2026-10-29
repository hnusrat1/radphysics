
def S_BS(T,I=75e-6):
    tau=T/mc2; b2=beta2(T)
    L=np.log(tau**2*(tau+2)/(2*(I/mc2)**2))
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/(tau+1)**2
    return 2*np.pi*re**2*mc2*ne/b2*(L+F)
for T in [0.1,0.5,1.0]:
    b2=beta2(T); Kc=2*np.pi*re**2*mc2*ne/b2
    print(T, Kc*J(T,75e-6,T/2), S_BS(T))
