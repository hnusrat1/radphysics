
import numpy as np, sys
sys.path.insert(0,'/tmp/work')
import mcph
print(mcph.kn_sample_mu(np.array([2.0,0.1]),np.array([0.5,0.5])))
p=np.array([[0,0,0.0],[0.5,0,0.0]]); d=np.array([[0,0,1.0],[0,0,1.0]])
print(mcph.chord_overlap(p,d,np.array([30.,30.])))
