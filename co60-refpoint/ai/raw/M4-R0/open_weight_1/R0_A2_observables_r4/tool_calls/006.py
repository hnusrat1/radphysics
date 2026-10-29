import numpy as np
from scipy.interpolate import interp1d

ph = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
Eph = ph[:,0]
coh = ph[:,1]; inc = ph[:,2]; pe = ph[:,3]; pair_n = ph[:,4]; pair_e = ph[:,5]; tot = ph[:,6]
def mklog(x,y):
    ok = (y>0)&(x>0)
    return interp1d(np.log(x[ok]), np.log(y[ok]), kind='linear', bounds_error=False, fill_value='extrapolate')
fcoh = mklog(Eph,coh); finc = mklog(Eph,inc); fpe = mklog(Eph,pe); fpn = mklog(Eph,pair_n); fpe2 = mklog(Eph,pair_e); ftot = mklog(Eph,tot)
def gx(f,E): return float(np.exp(f(np.log(float(E)))))

for E in [1.0,1.1732,1.3325]:
    c=gx(fcoh,E); i_=gx(finc,E); p=gx(fpe,E); pn=gx(fpn,E); pe_=gx(fpe2,E); t=gx(ftot,E)
    print(f"E={E}: coh={c:.5f} inc={i_:.5f} pe={p:.6f} pair_n={pn:.6f} pair_e={pe_:.6f} tot={t:.5f}")
    print("   frac: coh %.4f inc %.4f pe %.5f pair %.5f"%(c/t,i_/t,p/t,(pn+pe_)/t))

el = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=el[:,0]; Sc=el[:,1]; Sr=el[:,2]; St=el[:,3]; R=el[:,4]; Y=el[:,5]
print("\nESTAR:")
for E in [0.05,0.2,0.5,1.0,1.1732,1.3325,1.4]:
    print(f"E={E:.4f}: Sc={float(np.interp(E,Ee,Sc)):.4f} Sr={float(np.interp(E,Ee,Sr)):.5f} St={float(np.interp(E,Ee,St)):.4f} Y={float(np.interp(E,Ee,Y)):.5f}")