import numpy as np
# manually spectrum scaled to .075
v=np.array([.0001,.0002,.0003,.0005,.0007,.001,.0012,.0015,.0018,.0022,.0026,.003,.0035,.004,.0045,.005,.0055,.0055,.005,.0045,.004,.0035,.003,.0025,.0015,.001,.0008,.0005]);v*=.075/v.sum()
edge=np.arange(0,1.4001,.05); Ebin=(edge[:-1]+edge[1:])/2
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#'); logx=np.log(x[:,0])
def coeff(E,j): return np.exp(np.interp(np.log(E),logx,np.log(np.maximum(x[:,j],1e-300))))
def meanT(E):
 c=np.linspace(-1,1,10001); ep=1/(1+E/.51099895*(1-c)); ds=ep**2*(ep+1/ep-(1-c*c))/2; return np.trapezoid(ds*E*(1-ep),c)/np.trapezoid(ds,c)
# primary mean interactions coeff by lines; average fluence primary each .3645
pcoeff=np.mean([[coeff(E,j) for j in range(1,6)] for E in [1.1732,1.3325]],axis=0)
print('p coeff components coh,incoh,photo,pairN,pairE',pcoeff)
# rate vectors process coh/incoh/photo/pair
prate=np.array([pcoeff[0],pcoeff[1],pcoeff[2],pcoeff[3]+pcoeff[4]])*.729
srate=np.zeros(4); energytransfer=np.zeros(4)
for f,E in zip(v,Ebin):
 cs=np.array([coeff(E,j) for j in range(1,6)])
 rr=np.array([cs[0],cs[1],cs[2],cs[3]+cs[4]])*f
 srate+=rr
 energytransfer += np.array([0,rr[1]*meanT(E),rr[2]*E,rr[3]*max(E-1.022,0)])
# primary energy transfer rates
for E in [1.1732,1.3325]:
 cs=np.array([coeff(E,j) for j in range(1,6)])*.3645
 energytransfer += np.array([0,cs[1]*meanT(E),cs[2]*E,(cs[3]+cs[4])*max(E-1.022,0)])
rate=prate+srate
print('prate',prate,prate.sum(),prate/prate.sum())
print('srate',srate,srate.sum(),srate/srate.sum())
print('all frac',rate/rate.sum(), 'share scatter',srate.sum()/rate.sum())
print('transfer',energytransfer/energytransfer.sum(),energytransfer,energytransfer.sum())