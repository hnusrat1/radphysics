
import sys, importlib
sys.path.insert(0,'/tmp/work')
import deposit
importlib.reload(deposit)
for E in [2.0,1.0,0.5,0.1,0.05]:
    d=deposit.transfer_coeffs(np.array([E]))
    print(E, "K(tr)/E",d['tr'][0]/E, "Kcol/E",d['col'][0]/E)
