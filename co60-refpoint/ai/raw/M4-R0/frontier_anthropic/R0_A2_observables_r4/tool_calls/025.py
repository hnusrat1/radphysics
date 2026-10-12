
src=open('/tmp/work/ana3.py').read()
src=src.replace("    pz=sum(0.5*np.exp(-mu_tot(LN[i])*z)*(mu_inc(LN[i])*LN[i]*kcf_f(LN[i])+mu_pe(LN[i])*LN[i]*(1-Y_T(LN[i]))) for i in range(2))",
"    pz=(105.0/(100.0+z))**2*sum(0.5*np.exp(-mu_tot(LN[i])*z)*(mu_inc(LN[i])*LN[i]*kcf_f(LN[i])+mu_pe(LN[i])*LN[i]*(1-Y_T(LN[i]))) for i in range(2))")
src=src.replace("cavg=np.sum(Wp*cpc*Rp*(Tpc-0*Tpc))/np.sum(Wp*Rp)\nm1=0.4*cavg",
"cavg=np.sum(Wp*cpc*Rp)/np.sum(Wp)\nm1=0.45*cavg")
src=src.replace("print('PDD5 from grid'","# quadratic peak refinement\nzz=np.array(depth_list[:4]); yy=Kd[:4]\npc_=np.polyfit(zz[1:],yy[1:],2); zpk=-pc_[1]/(2*pc_[0]); print('peak depth est',zpk)\nprint('PDD5 from grid'")
open('/tmp/work/ana3.py','w').write(src)
import sys,time; sys.path.insert(0,'/tmp/work')
import importlib
import ana3
importlib.reload(ana3)
