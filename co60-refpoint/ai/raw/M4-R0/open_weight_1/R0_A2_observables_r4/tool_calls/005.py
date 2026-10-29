import numpy as np
from scipy.interpolate import interp1d

ph = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
Eph = ph[:,0]
coh = ph[:,1]; inc = ph[:,2]; pe = ph[:,3]; pair_n = ph[:,4]; pair_e = ph[:,5]; tot = ph[:,6]
def mklog(x,y):
    ok = (y>0)&(x>0)
    return interp1d(np.log(x[ok]), np.log(y[ok]), kind='linear', bounds_error=False, fill_value='extrapolate')
fcoh = mklog(Eph,coh); finc = mklog(Eph,inc); fpe = mklog(Eph,pe); fpn = mklog(Eph,pair_n); fpe2 = mklog(Eph,pair_e); ftot = mklog(Eph,tot)
def get(E):
    E=np.atleast_1d(np.asarray(E,float))
    return dict(E=E, coh=np.exp(fcoh(np.log(E))), inc=np.exp(finc(np.log(E))), pe=np.exp(fpe(np.log(E))),
                pair_n=np.exp(fpn(np.log(E))), pair_e=np.exp(fpe2(np.log(E))), tot=np.exp(ftot(np.log(E))))
for E in [1.0,1.1732,1.3325]:
    g=get(E)
    print(f"E={E}: coh={float(g['coh']):.5f} inc={float(g['inc']):.5f} pe={float(g['pe']):.6f} pair_n={float(g['pair_n']):.6f} pair_e={float(g['pair_e']):.6f} tot={float(g['tot']):.5f}")
    print("   fractions: coh %.4f inc %.4f pe %.5f pair %.5f"%(g['coh']/g['tot'],g['inc']/g['tot'],g['pe']/g['tot'],(g['pair_n']+g['pair_e'])/g['tot']))

el = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee=el[:,0]; Sc=el[:,1]; Sr=el[:,2]; St=el[:,3]; R=el[:,4]; Y=el[:,5]
print("\nESTAR sample:")
for E in [0.05,0.2,0.5,1.0,1.1732,1.3325,1.4]:
    sc=float(np.interp(E,Ee,Sc)); sr=float(np.interp(E,Ee,Sr)); st=float(np.interp(E,Ee,St)); yy=float(np.interp(E,Ee,Y))
    print(f"E={E:.4f}: Sc={sc:.4f} Sr={sr:.5f} St={st:.4f} Y={yy:.5f}")