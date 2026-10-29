import numpy as np
xcom = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x = xcom[:,0]; mu_coh=xcom[:,1]; mu_inc=xcom[:,2]; mu_pe=xcom[:,3]
mu_pair=xcom[:,4]+xcom[:,5]; mu_tot=xcom[:,6]
LE=np.log(E_x)
def mut(E, mu):
    mu=np.maximum(mu,1e-30)
    return np.exp(np.interp(np.log(E), LE, np.log(mu)))
E=0.11993051
mc=mut(E,mu_coh); mi=mut(E,mu_inc); mp=mut(E,mu_pe); mpr=mut(E,mu_pair); mt=mut(E,mu_tot)
print('coh,inc,pe,pair,tot=',mc,mi,mp,mpr,mt)
print('c1,c2,c3,c4=',mc, mc+mi, mc+mi+mp, mc+mi+mp+mpr)
print('diff tot-(c3)=', mt-(mc+mi+mp))
# Also print table rows near 0.1-0.15
m=(E_x>=0.08)&(E_x<=0.2)
print(xcom[m])