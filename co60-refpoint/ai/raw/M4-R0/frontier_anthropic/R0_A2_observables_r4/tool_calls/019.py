
import sys; sys.path.insert(0,'/tmp/work')
import pickle, numpy as np
SM=pickle.load(open('/tmp/work/SM.pkl','rb'))
Om=0.009975072683099257
print(SM['N'], SM['g']*Om*105**2/SM['N'])
print('hE per-phi0 (5cm):', SM['hE'][4]*Om*105**2/SM['N'])
