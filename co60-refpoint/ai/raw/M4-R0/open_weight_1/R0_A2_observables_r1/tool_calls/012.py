import numpy as np, math
mc2=0.5109989461
d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x=d[:,0]; mu_co=d[:,1]; mu_in=d[:,2]; mu_pe=d[:,3]; mu_pn=d[:,4]; mu_pef=d[:,5]; mu_tot=d[:,6]
def interp1(x,xx,yy): return np.interp(x,xx,yy)
def mu_tot_cm(E): return interp1(E,E_x,mu_tot)*1.0
re2 = 2.8179403262e-13**2  # cm^2
Ne = 3.343e23   # e/cm^3  (rho=1)
# numerical first-scatter integral on a grid
# P=(0,0,5). Q=(x,y,z). primary dir ~ zhat.
Es = [1.1732,1.3325]
ws = [0.5,0.5]
def integrand(E,z,x,y):
    # primary fluence per Phi0 at Q
    rp = math.sqrt(x*x+y*y+(100+z)**2)
    f_prim = (105.0/rp)**2 * math.exp(-mu_tot_cm(E)*z)
    R = math.sqrt(x*x+y*y+(5-z)**2)
    cos_s = (5-z)/R
    cs2 = cos_s*cos_s
    alpha=E/mc2
    xk = 1+alpha*(1-cos_s)
    Ep = E/xk
    dsdO = (re2/2.0)*(Ep/E)**2*((Ep/E)+(E/Ep)-(1-cs2))
    mu_sc = mu_tot_cm(Ep)
    return f_prim*Ne*dsdO*math.exp(-mu_sc*R)/R**2

tot=0.0
# integrate over pyramid: z 0..30, half width hw=min(15,5+0.05*z)
for E,w in zip(Es,ws):
    s=0.0
    nz=120; 
    for iz in range(nz):
        z=(iz+0.5)*30.0/nz
        hw=min(15.0,5+0.05*z)
        nx=40; ny=40
        for ix in range(nx):
            x=(-hw+ (ix+0.5)*2*hw/nx)
            for iy in range(ny):
                y=(-hw+(iy+0.5)*2*hw/ny)
                s+= integrand(E,z,x,y)*(2*hw/nx)*(2*hw/ny)*(30.0/nz)
    tot+=w*s
    print('line',E,'single scatter flu/Phi0 =',s)
print('total first-scatter flu/Phi0 =',tot)
print('MC scattered total was ~0.43 (includes multiple scatter)')