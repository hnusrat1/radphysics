import numpy as np
# recalc electron source hist by source rates per Phi0 using KN and interaction rates; primary attenuated lines rates, scat spectrum event rate
m=.51099895; ee=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4]); mids=(ee[:-1]+ee[1:])/2
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#');lx=np.log(x[:,0]);
def ci(E,j):return np.exp(np.interp(np.log(E),lx,np.log(np.maximum(x[:,j],1e-300))))
ph_edges=np.linspace(0,1.4,29);pe=(ph_edges[:-1]+ph_edges[1:])/2
vf=np.array([.0001,.0002,.0003,.0005,.0007,.001,.0012,.0015,.0018,.0022,.0026,.003,.0035,.004,.0045,.005,.0055,.0055,.005,.0045,.004,.0035,.003,.0025,.0015,.001,.0008,.0005]);vf*=.075/vf.sum()
cc=np.linspace(-1,1,40001);dcc=cc[1]-cc[0]
def hst(E, weight):
 ep=1/(1+E/m*(1-cc));ds=ep**2*(ep+1/ep-(1-cc**2))/2;T=E*(1-ep);w=ds*dcc;w[[0,-1]]*=.5;w/=w.sum(); mask=T>=.01
 return np.histogram(T[mask],ee,weights=weight*w[mask])[0]
src=np.zeros(25)
for E in [1.1732,1.3325]: src+=hst(E,.3645*ci(E,2))
for E,f in zip(pe,vf): src+=hst(E,f*ci(E,2))
print('source',src.sum(),src@ mids/src.sum())
# survival source above midpoint
estar=np.loadtxt('/tmp/data/estar_water.txt',comments='#'); S=np.interp(mids,estar[:,0],estar[:,3]); surv=np.array([src[mids>=e].sum() for e in mids]); raw=surv/S*np.diff(ee)
print('base fluence per phi0',sum(raw),raw/sum(raw), 'mean E',np.dot(raw,mids)/sum(raw))
print('range avg effective',sum(raw)/src.sum())