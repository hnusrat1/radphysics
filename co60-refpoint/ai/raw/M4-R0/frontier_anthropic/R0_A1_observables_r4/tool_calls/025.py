
import pickle, json, numpy as np
out=pickle.load(open('/tmp/work/final.pkl','rb'))
def r(x):
    if isinstance(x,list): return [float('%.5g'%v) for v in x]
    if isinstance(x,dict): return {k:float('%.5g'%v) for k,v in x.items()}
    return float('%.5g'%x)
o={k:(r(v) if k not in ('method','data_sources') else v) for k,v in out.items()}
o['L5_beta']=1.0
print(json.dumps(o,separators=(',',':')))
