
import time, sys
sys.path.insert(0,'.')
import importlib, mc2
importlib.reload(mc2)
t=time.time()
depths=[0.5,5.0]
res=mc2.run(200000,depths,seed=2)
print("time",time.time()-t, res['nsteps'])
tal=res['tal']
V=mc2.Vcyl; P0=mc2.Phi0_hist
for c in range(2):
    print("depth",depths[c])
    print(" primary fluence/phi0", tal['Lp'][c]/(200000*V*P0), " total", (tal['Lp'][c]+tal['Ls'][c])/(200000*V*P0))
    print(" interactions C,PE,PP,R (p/s):", [round(tal[f'I{x}_p'][c]+tal[f'I{x}_s'][c],6) for x in ['C','PE','PP','R']])
