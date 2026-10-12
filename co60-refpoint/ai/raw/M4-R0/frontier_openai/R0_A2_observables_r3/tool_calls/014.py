import numpy as np
m=.51099895
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
v=np.array([.0001,.0002,.0003,.0005,.0007,.001,.0012,.0015,.0018,.0022,.0026,.003,.0035,.004,.0045,.005,.0055,.0055,.005,.0045,.004,.0035,.003,.0025,.0015,.001,.0008,.0005]);v*=.075/v.sum(); epht=np.linspace(0,1.4,29);Ebin=(epht[:-1]+epht[1:])/2
x=np.loadtxt('/tmp/data/xcom_water.txt',comments='#'); logx=np.log(x[:,0]);
def ci(E,j): return np.exp(np.interp(np.log(E),logx,np.log(np.maximum(x[:,j],1e-300))))
cc=np.linspace(-1,1,40001); dcc=cc[1]-cc[0]
def rh(E, weight, hist):
 ep=1/(1+E/m*(1-cc)); ds=ep**2*(ep+1/ep-(1-cc**2))/2; T=E*(1-ep); ww=ds*dcc;ww[[0,-1]]*=.5; norm=ww.sum();mask=T>=.01
 hist+=np.histogram(T[mask],bins=edges,weights=weight*ww[mask]/norm)[0]
 return weight*np.sum(T[mask]*ww[mask])/norm,weight*np.sum(ww[mask])/norm
hp=np.zeros(25);hs=np.zeros(25);mp=cp=ms=cs=0
for E in [1.1732,1.3325]:
 w=.3645*ci(E,2); a,b=rh(E,w,hp);mp+=a;cp+=b
for E,f in zip(Ebin,v):
 w=f*ci(E,2);a,b=rh(E,w,hs);ms+=a;cs+=b
q=hp+hs
print(cp,cs,cs/(cp+cs),mp/cp,ms/cs, (q@((edges[:-1]+edges[1:])/2)/q.sum()))
print('all', (q/q.sum()).tolist())
print('parentprimary', (hp/hp.sum()).tolist())