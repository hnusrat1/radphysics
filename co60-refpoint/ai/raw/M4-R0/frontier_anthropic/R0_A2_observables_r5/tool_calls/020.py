
ana = r'''
import numpy as np, pickle
from mcana import *
from mcmod import *

def primary_fluence_per_phi0():
    z=np.linspace(4.9,5.1,2001)
    out=[]
    for E in LINES:
        mu=mu_tot(E)
        out.append(0.5*np.mean(np.exp(-mu*z)*(105.0/(100.0+z))**2))
    return np.array(out)   # per Phi0, per line (each 0.5 share)

def primary_electron_tally(rng, Ns=400000):
    tal=new_tally()
    P=primary_fluence_per_phi0()*Phi0_src    # per source photon, per line
    for i,E in enumerate(LINES):
        Ep=np.full(Ns,E); om=np.tile([0,0,1.0],(Ns,1)); val=np.full(Ns,P[i]/Ns)
        lst=electrons_from_arrivals(Ep,om,val,rng,True)
        # tally into 'prim' dict
        for (T,cz,w,ispos) in lst:
            tal['el_w_all']+=w.sum(); tal['el_wT_all']+=(w*T).sum()
            h,_=np.histogram(T,bins=EL_EDGES,weights=w); tal['el_T_all']+=h
            hc,_=np.histogram(cz,bins=CO_EDGES,weights=w); tal['el_C_all']+=hc
            ok=(T>=FINE[0])&(T<FINE[-1]); fb=np.clip(np.digitize(T,FINE)-1,0,NFINE-1); cb=np.clip(np.digitize(cz,CO_EDGES)-1,0,9)
            np.add.at(tal['q_fine_all'],fb[ok],w[ok]); np.add.at(tal['q_fine_all_cos'],(fb[ok],cb[ok]),w[ok])
            m=ok&(~ispos)
            np.add.at(tal['q_fine_em'],fb[m],w[m]); np.add.at(tal['q_fine_em_cos'],(fb[m],cb[m]),w[m])
    return tal, P
'''
open('/tmp/work/ana.py','w').write(ana)
import sys, time, pickle; sys.path.insert(0,'/tmp/work')
import numpy as np, importlib
import mcana, mcmod, ana as A
importlib.reload(mcana); importlib.reload(A)
from mcana import *
tal=pickle.load(open('/tmp/work/tal_run1.pkl','rb'))
rng=np.random.default_rng(7)
ptal,Pper_src = A.primary_electron_tally(rng)
print("primary per Phi0 per line", A.primary_fluence_per_phi0())
print("P sums", Pper_src/Phi0_src)
print("primary electron total weight per src", ptal['el_w_all'], "vs primary int mu_inc*phi", )
