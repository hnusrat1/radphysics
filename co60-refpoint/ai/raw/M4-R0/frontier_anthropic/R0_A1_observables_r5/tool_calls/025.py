
dbg2 = r'''
import numpy as np, pickle, sys
sys.path.insert(0,'/tmp/work')
import mc, phys, ea
from phys import me
seg,hist,NTOT=pickle.load(open('/tmp/work/tallies.pkl','rb'))
V=np.pi*1.0**2*0.2
R=mc.Omega*105.0**2/(V*NTOT)
E=seg['E']; tl=seg['tl']
w=tl*R
muC=mc.mu_c(E)
# check: sum w*muC, and mean Compton T from KN analytic: <T>=E*(sigma_tr/sigma)
from scipy import integrate
Eg=np.linspace(0.0005,1.45,3000)
mutr=np.array([phys.ne_w*integrate.quad(lambda c: phys.dsig_dOmega(e,c)*2*np.pi*e*(1-1/(1+e/me*(1-c))),-1,1)[0] for e in Eg])
print('K from interp', (w*np.interp(E,Eg,mutr)).sum())
print('sum w muC', (w*muC).sum(), ' sum w muC*<T>/E', (w*np.interp(E,Eg,mutr)).sum())
'''
open('/tmp/work/dbg2.py','w').write(dbg2)
o,e=runpy('/tmp/work/dbg2.py'); print(o,e)
