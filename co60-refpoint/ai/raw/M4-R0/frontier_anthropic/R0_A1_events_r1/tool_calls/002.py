
for E in [0.1,0.2,0.5,1.0,1.1732,1.25,1.3325]:
    print(E,[float(np.round(x,6)) for x in mu_parts(np.array(E))])
for E in Es:
    a=E/me; c=np.cos(ths); Ep=E/(1+a*(1-c)); eps=Ep/E
    f=eps**2*(eps+1/eps-np.sin(ths)**2)
    print(E,f.max())
