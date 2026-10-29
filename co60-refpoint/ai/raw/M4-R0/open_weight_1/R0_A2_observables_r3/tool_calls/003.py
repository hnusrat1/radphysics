import numpy as np
d2 = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=d2[:,0]; Sc=d2[:,1]; Sr=d2[:,2]; St=d2[:,3]; R=d2[:,4]; Y=d2[:,5]; de=d2[:,6]

me=0.51099895
re2=(2.8179403e-13)**2
NA=6.02214076e23
C=2*np.pi*re2*me*NA  # MeV cm^2 / mol

def bethe(Ek, Z, A, I_ev, delta=0.0):
    tau=Ek/me; g=tau+1; b2=1-1/g**2
    Im= I_ev*1e-3/me
    bracket = np.log(tau**2*(tau+2)/(2*Im**2)) + 1 - b2 + (tau**2/8 - (2*tau+1)*np.log(2))/(g**2) - delta
    return C*(Z/A)/b2*bracket

Etest=[0.1,0.2,0.5,1.0,1.3325,2.0,5.0,10.0]
for e in Etest:
    i=np.argmin(np.abs(Ee-e))
    Sc_est=Sc[i]; delta=de[i]
    Sc_b=bethe(e,10,18.015268,75.0,delta)
    print(f"E={e:5.2f}: ESTAR={Sc_est:7.4f} Bethe={Sc_b:7.4f} ratio={Sc_est/Sc_b:5.3f} delta={delta:.4f}")

# check with I=78, no delta
print()
for I_ev in [75.0]:# ,78.0]:
    for e in [0.1,1.0,1.3325]:
        Sc_b=bethe(e,10,18.015268,I_ev,0.0)
        i=np.argmin(np.abs(Ee-e))
        print(f"I={I_ev} E={e}: Bethe(no d)={Sc_b:.4f} ESTAR={Sc[i]:.4f}")