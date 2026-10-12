
s=open('/tmp/work/mell.py').read()
s+=r'''
# hard (delta>DCUT) collisional loss per cm for electrons of energy T (MeV/cm, water)
HG=np.exp(np.linspace(np.log(0.02),np.log(2.0),200))
HV=np.zeros(len(HG))
for i,T in enumerate(HG):
    tau=T/ME; b2=1-1/(1+tau)**2
    if DCUT/T>=0.5: HV[i]=0; continue
    x=np.linspace(DCUT/T,0.5,20001)
    HV[i]=K0*0.55509/b2*np.trapezoid(x*Fx(x,tau),x)
def Hloss(T):
    T=np.asarray(T,float)
    return np.interp(np.log(np.clip(T,HG[0],HG[-1])),np.log(HG),HV)*(T>=2*DCUT)
'''
open('/tmp/work/mell.py','w').write(s)
e=open('/tmp/work/elec.py').read()
e=e.replace("""        depp=np.clip(Sc/St*dT-epsum,0,None)
        dep_tot+=depp.sum()""","""        Hl=mell.Hloss(T)
        Scr=np.maximum(Sc-Hl,0.0)
        depp=Scr*ds
        dep_tot+=depp.sum()""")
e=e.replace("""        Tn=T-dT
        c=cn""","""        Tn=T-((Scr+Sr)*ds+epsum)
        c=cn""")
open('/tmp/work/elec.py','w').write(e)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, numpy as np, elec, mell
print(mell.Hloss(np.array([0.1,1.0,2.0])), mell.Hloss(1.0)/xs.Scol(1.0) if False else '')
B=np.array([[1.0,1.0,0,1.0,0]])
r=elec.run_e(B,1.0,4000,3)
print('dep',r['dep_tot']/4000, 'expected', 1-0.00358, 'fl',r['fl_tot'], 'del frac',r['fl_del']/r['fl_tot'], r['steps'])
