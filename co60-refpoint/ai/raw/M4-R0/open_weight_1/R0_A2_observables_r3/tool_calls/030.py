import numpy as np
me=0.51099895
r0=2.8179403e-13; NA=6.02214076e23
C=2*np.pi*r0**2*me*NA
d2=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
_Ee=d2[:,0]; _Sc=d2[:,1]; _de=d2[:,6]

def bethe_col(Ek,Z,A,I_ev,delta):
    tau=Ek/me; g=tau+1; b2=1-1/g**2
    Im=I_ev*1e-3/me
    br=np.log(tau**2*(tau+2)/(2*Im**2)) + 1 - b2 + (tau**2/8 - (2*tau+1)*np.log(2))/(g**2) - delta
    return C*(Z/A)/b2*br

ZA_w=10/18.015268
for e in [0.05,0.1,0.2,0.5,1.0,1.33,2.0]:
    i=np.argmin(abs(_Ee-e))
    b=bethe_col(_Ee[i],10,18.015268,75.0,_de[i])
    print(f"E={_Ee[i]:.3f}: ESTAR={_Sc[i]:.5f} Bethe(with delta)={b:.5f} ratio={_Sc[i]/b:.4f}")
# air params: composition
# N2 0.78084, O2 0.20946, Ar 0.00934 (mol fractions)
# mass fractions: N2:0.7553,O2:0.2315,Ar:0.0132
ZA_air=0.7553*(14/28.0134)+0.2315*(16/31.998)+0.0132*(18/39.948) # = sum w_i Z_i/A_i
print("ZA_air:",ZA_air," Ne_air:",NA*ZA_air)
# air collision SP at selected energies with I=85.7 eV, delta=0
for e in [0.05,0.1,0.2,0.5,1.0,1.33,2.0,10.0]:
    b=bethe_col(e,1,1/ZA_air,85.7,0.0)  # Z/A passed as ratio; careful: function uses (Z/A)
    print(f"air E={e}: S_col={b:.5f} MeV cm2/g")
# Known ESTAR air values ~ at 1 MeV 1.645? print