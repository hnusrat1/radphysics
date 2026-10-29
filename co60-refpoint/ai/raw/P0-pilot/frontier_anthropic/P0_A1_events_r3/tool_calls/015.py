
import numpy as np, pickle, json
ev=pickle.load(open('events_6000000.pkl','rb'))
rng2=np.random.default_rng(2024)
idx=rng2.choice(len(ev),200,replace=False)
evs=[]
for i in idx:
    proc,ein,eout,cs,te,ce,tp=ev[i]
    r=lambda x: None if x is None else float(f"{x:.4g}")
    evs.append({"process":proc,"e_in_MeV":float(f"{ein:.4g}"),"e_out_MeV":r(eout),"cos_scatter":r(cs),
                "t_e_MeV":float(f"{te:.4g}"),"cos_electron":r(ce),"t_positron_MeV":r(tp)})
json.dump(evs,open('final_events.json','w'))
print(len(evs))
for k in range(0,200,50):
    print(json.dumps(evs[k:k+50],separators=(',',':')))
    print("CHUNK_END",k)
