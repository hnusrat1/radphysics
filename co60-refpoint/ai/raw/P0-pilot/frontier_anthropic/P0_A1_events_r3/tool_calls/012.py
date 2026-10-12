
import numpy as np, pickle
ev=pickle.load(open('events_6000000.pkl','rb'))
rng2=np.random.default_rng(2024)
idx=rng2.choice(len(ev),200,replace=False)
rows=[]
for i in idx:
    proc,ein,eout,cs,te,ce,tp=ev[i]
    f=lambda x: 'null' if x is None else f"{x:.4g}"
    rows.append(f'["{proc}",{ein:.4g},{f(eout)},{f(cs)},{te:.4g},{f(ce)},{f(tp)}]')
print("\n".join(rows[70:140]))
