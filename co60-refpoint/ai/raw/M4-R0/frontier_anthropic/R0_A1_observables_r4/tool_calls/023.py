
chk = r'''
import sys; sys.path.insert(0,'/tmp/work')
import pickle, numpy as np, mc as MC
Tc,kerma,flu,ev,Ntot=pickle.load(open('/tmp/work/comb.pkl','rb'))
VOL=MC.VOL; F0=MC.F0; conv=1.602176634e-10
norm=F0/(Ntot*VOL)
Kp=(Tc['trp_C']+Tc['trp_pe']+Tc['trp_pair'])*norm*conv
Ks=(Tc['trs_C']+Tc['trs_pe']+Tc['trs_pair'])*norm*conv
print('K_prim(5)',Kp,'K_scat(5)',Ks,'scat fraction of kerma',Ks/(Kp+Ks),'s/p',Ks/Kp)
print('K total(5) from cyl',kerma[0]/Ntot*norm*conv,' K(0.5)',kerma[1]/Ntot*norm*conv)
print('primary-only K(0.5) predicted',Kp/0.6895)
print('K(0.5)/K_prim(0.5) implied scatter ratio', kerma[1]/Ntot*norm*conv/(Kp/0.6895)-1)
'''
open('/tmp/work/chk.py','w').write(chk)
import subprocess
r=subprocess.run(['python','/tmp/work/chk.py'],capture_output=True,text=True,cwd='/tmp/work'); print(r.stdout, r.stderr[-1500:])
