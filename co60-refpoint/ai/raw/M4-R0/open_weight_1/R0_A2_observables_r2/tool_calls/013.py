import numpy as np
# minimal test of np.add.at with 4D-style indexing (3 indices)
score = np.zeros((28,2,10))
Eb = np.array([23,23]); pr=np.array([0,0]); cb=np.array([9,9]); contrib=np.array([0.5,0.3])
np.add.at(score,(Eb,pr,cb),contrib)
print(score[:,0,:].sum(), score[23,0,9])

# test searchsorted for Ea
Ea = np.array([1.1732,1.3325])
ph_edges = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ct_edges = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
print(np.searchsorted(ph_edges,Ea,side='right')-1)
print(np.searchsorted(ct_edges,np.array([1.0]),side='right')-1)