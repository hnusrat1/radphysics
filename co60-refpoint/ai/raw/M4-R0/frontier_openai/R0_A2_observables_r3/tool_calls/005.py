import numpy as np
m=.51099895
edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
ce=np.linspace(-1,1,1000001)
weights=[]; energies=[]; mus=[]
for E in [1.1732,1.3325]:
 eps=1/(1+E/m*(1-ce)); ds=eps**2*(eps+1/eps-(1-ce**2))/2
 T=E*(1-eps)
 # electron angle cosine from momentum conservation: p_e cos = E - E' c; p=sqrt(T(T+2m))
 mu=(E-Eps if False else E - E*eps*ce)/np.sqrt(np.maximum(T*(T+2*m),1e-30))
 mask=T>=.01
 # integration over c via bin weighted approx
 weights.append(ds[mask]); energies.append(T[mask]); mus.append(mu[mask])
# compute bin with trapezoidal quadrature point arrays
Earr=np.concatenate(energies); muarr=np.concatenate(mus); warr=np.concatenate(weights)
# equal line but differential cross sect total differs, need line interaction counts same photons, electron interactions cross-sections nearly same slight; normalization equal line counts and interactions probabilities. normalized by KN integrated; per line normalize each
warr=np.concatenate([weights[0]/np.trapezoid(weights[0],ce[ (1/(1+1.1732/m*(1-ce)))<1]) if False else weights[0]])
# easier direct each line differential distribute trapezoid weights, normalize line
hist=np.zeros(len(edges)-1); ang=np.zeros(10); mean_num=den=0
for E in [1.1732,1.3325]:
 eps=1/(1+E/m*(1-ce)); ds=eps**2*(eps+1/eps-(1-ce**2))/2; T=E*(1-eps)
 mu=(E-E*eps*ce)/np.sqrt(np.maximum(T*(T+2*m),1e-30))
 w=ds.copy(); w[0]*=.5;w[-1]*=.5; w*=2/len(ce)
 mask=T>=.01
 hist += np.histogram(T[mask],edges,weights=w[mask]/2)[0]/np.sum(w[mask])/2
 ang += np.histogram(np.clip(mu[mask],-1,1),np.linspace(-1,1,11),weights=w[mask]/2)[0]/np.sum(w[mask])/2
 mean_num += np.sum(T[mask]*w[mask])/np.sum(w[mask])/2
 den+=.5
print(hist, hist.sum(), mean_num/den)
print(ang,ang.sum())