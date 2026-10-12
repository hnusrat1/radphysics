
import sys, os
sys.path.insert(0,'/tmp/work')
import pickle, numpy as np
from phys import *
trap=np.trapezoid
outs=[pickle.load(open(f'/tmp/work/out_{s}.pkl','rb')) for s in [11,12,13,14]]
Nt=sum(o['N'] for o in outs)
Omega=outs[0]['Omega']; Vc=outs[0]['Vc']
NORM=Omega*110.0**2/(Nt*Vc)
inter={k:sum(o['inter'][k] for o in outs) for k in outs[0]['inter']}
transf={k:sum(o['transf'][k] for o in outs) for k in outs[0]['transf']}
Lsum={p:sum(o['Lsum'][p] for o in outs) for p in 'PS'}
ESum={p:sum(o['ESum'][p] for o in outs) for p in 'PS'}
hp=sum(o['hist_prim'] for o in outs); hs=sum(o['hist_scat'] for o in outs)
ap=sum(o['ang_prim'] for o in outs); as_=sum(o['ang_scat'] for o in outs)
ep={k:np.concatenate([o['ep'][k] for o in outs]) for k in outs[0]['ep']}
ev={k:np.concatenate([o['ev'][k] for o in outs]) for k in outs[0]['ev']}
print("Nt",Nt,"events in r<3",len(ev['T']))
fP=NORM*Lsum['P']; fS=NORM*Lsum['S']
print("L1 primary/phi0",fP,"scattered/phi0",fS,"total",fP+fS,"scat frac",fS/(fP+fS))
print("scat energy fraction", ESum['S']/(ESum['P']+ESum['S']))
print("mean photon energy", (ESum['P']+ESum['S'])/(Lsum['P']+Lsum['S']))
print("ang primary", ap/ap.sum()); print("ang scat", as_/as_.sum())
print("scattered spectrum/phi0:", np.round(NORM*hs,5))
