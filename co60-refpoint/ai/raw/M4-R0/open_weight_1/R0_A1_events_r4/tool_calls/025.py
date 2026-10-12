import math

E=1.1732
mec2=0.511
a=E/mec2

for cos_theta in [0.9, 0.5, 0.0, -0.5, -0.9]:
    k=1.0/(1.0+a*(1.0-cos_theta))
    Eout=k*E
    te=E-Eout
    theta=math.acos(cos_theta)
    tan_half=math.tan(theta/2)
    cot_phi=(1.0+a)*tan_half
    phi=math.atan2(1.0,cot_phi)
    pe=math.sqrt(te*(te+2*mec2))
    if pe>0:
        cos_phi2=(E-Eout*cos_theta)/pe
    else:
        cos_phi2=1.0
    print(f"cos_theta={cos_theta:.3f}, Eout={Eout:.4f}, te={te:.4f}, phi={math.degrees(phi):.2f}deg, cos_phi={cos_phi2:.6f}, my_cos={math.cos(phi):.6f}")