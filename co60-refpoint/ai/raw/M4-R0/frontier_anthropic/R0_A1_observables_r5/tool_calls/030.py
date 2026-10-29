
chk = r'''
import numpy as np, pickle, sys
sys.path.insert(0,'/tmp/work')
d=pickle.load(open('/tmp/work/part2.pkl','rb'))
import mc
seg,hist,NTOT=pickle.load(open('/tmp/work/tallies.pkl','rb'))
V=np.pi*1.0**2*0.2; R=mc.Omega*105.0**2/(V*NTOT)
E=seg['E']; tl=seg['tl']; w=tl*R
Phi=d['Phi']; Tg=d['Tg']; dT=0.001
m=Tg>=0.01
Sc=d['Scol']; Sr=d['Srad']
print('K (MeV/cm per Phi0) from photons: ', (w*0+0).sum())
from scipy import integrate
import phys
Eg=np.linspace(0.0005,1.45,3000)
mutr=np.array([phys.ne_w*integrate.quad(lambda c: phys.dsig_dOmega(e,c)*2*np.pi*e*(1-1/(1+e/phys.me*(1-c))),-1,1)[0] for e in Eg])
mutrE=np.interp(E,Eg,mutr)
mupe=mc.mu_pe(E); mupair=mc.mu_pair(E)
from phys import me
K=(w*(mutrE+mupe*E+mupair*(E-2*me))).sum()
print('K',K)
print('int Phi*Scol (unrestricted, >=10keV)',(Phi*Sc)[m].sum()*dT)
print('int Phi*Stot_restricted (>=10keV)',(Phi*(Sc+Sr))[m].sum()*dT)
print('int Phi*Srad (>=10keV)',(Phi*Sr)[m].sum()*dT)
print('Phi total',Phi[m].sum()*dT)
'''
open('/tmp/work/chk.py','w').write(chk)
import subprocess, sys
def runpy(path):
    r=subprocess.run([sys.executable,path],capture_output=True,text=True,cwd='/tmp/work')
    return r.stdout, r.stderr[-3000:]
o,e=runpy('/tmp/work/chk.py'); print(o,e)
