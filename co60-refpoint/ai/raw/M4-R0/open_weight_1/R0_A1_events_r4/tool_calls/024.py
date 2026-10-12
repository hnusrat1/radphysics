import math

# Verify Compton kinematics
E=1.1732
mec2=0.511
a=E/mec2

# Test a few angles
for cos_theta in [1.0, 0.9, 0.5, 0.0, -0.5, -0.9]:
    k=1.0/(1.0+a*(1.0-cos_theta))
    Eout=k*E
    te=E-Eout
    theta=math.acos(cos_theta)
    # Electron angle from cot formula
    tan_half=math.tan(theta/2)
    cot_phi=(1.0+a)*tan_half
    phi=math.atan2(1.0, cot_phi)
    # Also compute from momentum conservation
    # Electron momentum: p_e = sqrt(te*(te+2*mec2)) / c (in MeV/c units)
    # Photon momentum in: p_in = E/c
    # Photon momentum out: p_out = Eout/c
    # cos_phi = (p_in - p_out*cos_theta) / p_e
    pe = math.sqrt(te*(te+2*mec2))
    cos_phi2 = (E - Eout*cos_theta) / pe
    print(f"cos_theta={cos_theta:.3f}, theta={theta:.4f}, phi={phi:.4f}, cos_phi={cos_phi2:.6f}, my_cos_phi={math.cos(phi):.6f}")