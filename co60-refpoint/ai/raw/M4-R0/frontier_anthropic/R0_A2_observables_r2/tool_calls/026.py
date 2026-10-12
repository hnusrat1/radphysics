
s=open('/tmp/ana4.py').read()
s=s.replace("def kn_hist(E):\n    g,w=np.polynomial.legendre.leggauss(3000)",
            "_KG,_KW=np.polynomial.legendre.leggauss(1500)\ndef kn_hist(E):\n    g,w=_KG,_KW")
s=s.replace("Pmat=np.zeros((M.NEB,nT))\nfor j in range(M.NEB):\n    if EC[j]>0.0011: Pmat[j]=kn_hist(EC[j])",
"import os\nif os.path.exists('/tmp/cache_P.npy'):\n    Pmat=np.load('/tmp/cache_P.npy')\nelse:\n    Pmat=np.zeros((M.NEB,nT))\n    for j in range(M.NEB):\n        if EC[j]>0.0011: Pmat[j]=kn_hist(EC[j])\n    np.save('/tmp/cache_P.npy',Pmat)")
open('/tmp/ana4.py','w').write(s)
import time
t0=time.time()
g={}
exec(s,g)
print('done',time.time()-t0)
print(g['chk'])
