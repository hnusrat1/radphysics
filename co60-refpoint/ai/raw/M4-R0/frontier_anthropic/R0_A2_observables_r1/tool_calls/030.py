
a=open('/tmp/work/ana.py').read()
a=a.replace("""def electron_sources(Wcells, Wprim):""","""def Rng(T):
    return np.exp(np.interp(np.log(np.maximum(T,1e-4)),np.log(ST[:,0]),np.log(ST[:,4])))
def electron_sources(Wcells, Wprim):""")
a=a.replace("""        hc,_=np.histogram(np.clip(cc,-1,1),bins=Cedges,weights=ww)
        if key not in out: out[key]=[np.zeros(700),np.zeros(10),0.0,0.0,0.0,0.0]
        out[key][0]+=hT; out[key][1]+=hc""","""        hc,_=np.histogram(np.clip(cc,-1,1),bins=Cedges,weights=ww)
        hR,_=np.histogram(np.clip(cc,-1,1),bins=Cedges,weights=ww*Rng(T))
        if key not in out: out[key]=[np.zeros(700),np.zeros(10),np.zeros(10),0.0,0.0,0.0]
        out[key][0]+=hT; out[key][1]+=hc; out[key][2]+=hR""")
a=a.replace("""        hc,_=np.histogram(np.full(len(Tarr),cval),bins=Cedges,weights=weight)
        if key not in out: out[key]=[np.zeros(700),np.zeros(10),0.0,0.0,0.0,0.0]
        out[key][0]+=hT; out[key][1]+=hc""","""        hc,_=np.histogram(np.full(len(Tarr),cval),bins=Cedges,weights=weight)
        hR,_=np.histogram(np.full(len(Tarr),cval),bins=Cedges,weights=weight*Rng(Tarr))
        if key not in out: out[key]=[np.zeros(700),np.zeros(10),np.zeros(10),0.0,0.0,0.0]
        out[key][0]+=hT; out[key][1]+=hc; out[key][2]+=hR""")
open('/tmp/work/ana.py','w').write(a)
m=open('/tmp/work/main.py').read()
m=m.replace("out['L4_angular']=(Ctot/Ctot.sum()).tolist()","CRtot=src['prim'][2]+src['scat'][2]\nout['L4_angular']=(CRtot/CRtot.sum()).tolist()")
open('/tmp/work/main.py','w').write(m)
p=subprocess.run([sys.executable,'/tmp/work/main.py'],capture_output=True,text=True,cwd='/tmp/work')
print(p.returncode,p.stderr[-1500:])
R=json.load(open('/tmp/work/results.json'))
print(R['L4_angular'], R['L4_electron_fluence_per_phi0'], R['L5_s_w_air'])
