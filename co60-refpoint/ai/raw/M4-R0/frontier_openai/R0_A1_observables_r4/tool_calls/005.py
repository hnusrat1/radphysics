import numpy as np
E=1.3325; mu=np.linspace(-1,1,10001); al=E/.51099895; ep=1/(1+al*(1-mu)); T=E*(1-ep); print(T.max(),T.min(),T[T>1].size,T.max())