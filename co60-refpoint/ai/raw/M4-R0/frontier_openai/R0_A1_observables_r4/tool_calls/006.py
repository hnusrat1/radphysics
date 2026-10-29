# diagnostic calc
from numpy.polynomial.legendre import leggauss
x,gw=leggauss(1000);mu=(x+1)/2;gw/=2;e=1.3325; alpha=e/.51099895; ep=1/(1+alpha*(1-mu));T=e*(1-ep);kn=ep**2*(ep+1/ep-(1-mu**2));print(T.max(),sum(gw[ T>1]*kn[T>1]),sum(gw*kn),np.histogram(T,bins=edges,weights=gw*kn)[0][-7:])