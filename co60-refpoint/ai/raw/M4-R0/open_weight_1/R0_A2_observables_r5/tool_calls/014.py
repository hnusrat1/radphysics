import numpy as np
# solid angle of pyramid with square cross-section 10x10 at distance 100 (half-angle alpha=beta)
alpha = np.arctan(5.0/100.0)
Omega = 4*np.arcsin(np.sin(alpha)*np.sin(alpha))
print('Omega', Omega, 'approx 0.01', Omega-0.01)
d = 105.0
Phi0_per_emitted = 1.0/(Omega*d*d)
print('Phi0 per emitted (cm^-2)', Phi0_per_emitted)
# analytic primary fluence ratio
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x=xcom[:,0]; mu_tot=xcom[:,6]; LE=np.log(E_x)
def mut(E):
    return np.exp(np.interp(np.log(E), LE, np.log(mu_tot)))
mu1=mut(1.1732); mu2=mut(1.3325)
print('mu1,mu2', mu1, mu2)
ratio = 0.5*(np.exp(-mu1*5)+np.exp(-mu2*5))
print('analytic L1_primary_fluence_per_phi0=', ratio)
V = np.pi*1.0**2*0.2
fl_prim=205.793368478072; N=50000
simratio = fl_prim/(N*V)/Phi0_per_emitted
print('sim ratio =', simratio)