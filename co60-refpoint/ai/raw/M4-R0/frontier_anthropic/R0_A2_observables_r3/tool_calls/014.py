
dbg = r'''
import sys, pickle, numpy as np
sys.path.insert(0,'/tmp/work')
import phys, mc
from phys import *
R = pickle.load(open('/tmp/work/res_t1.pkl','rb'))
f = 1.25e6/2e7
T = R['birth_T']; w = R['birth_w']*f; pr=R['birth_proc']; sg=R['birth_sign']
print('births per Phi0 (corr):', w.sum(), ' sum wT/V', np.sum(w*T)/mc.V_POI, ' mean T', np.sum(w*T)/w.sum())
for p in range(3):
    m = pr==p
    print('proc',p, 'w', w[m].sum(), 'meanT', np.sum(w[m]*T[m])/w[m].sum())
print('cnt inc corr', R['cnt']['inc']*f, 'etr inc corr', R['etr']['inc']*f)
print('sum wT vs etr:', np.sum(w*T), R['etr']['inc']*f+R['etr']['pe']*f+R['etr']['pair']*f)
print('Tsc (sum wl)', R['Tsc'].sum(), ' Tsc*PREF/V', R['Tsc'].sum()*mc.OMEGA*mc.DIST**2/2e7/mc.V_POI)
print('Esum_w corr/V', R['Esum_w']*f/mc.V_POI)
'''
open('/tmp/work/dbg.py','w').write(dbg)
import subprocess
r=subprocess.run("cd /tmp/work && python dbg.py",shell=True,capture_output=True,text=True)
print(r.stdout, r.stderr[-2000:])
