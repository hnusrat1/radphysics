
import sys; sys.path.insert(0,'/tmp/work')
import mc as MC, mut, numpy as np, time, pickle
cos_edges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
Eph=np.arange(0,1.4001,0.05)
t0=time.time()
res=[]
NB=60
for s in range(NB):
    res.append(MC.run(250000,1000+s,mut.mu_fun,[105.0,100.5],Eph,cos_edges))
print('time',time.time()-t0)
# combine
Ntot=250000*NB
def comb(res):
    T0=res[0][0]; 
    Tc={k:(sum(r[0][k] for r in res)) for k in res[0][0]}
    kerma=sum(r[1] for r in res); flu=sum(r[2] for r in res)
    ev=np.concatenate([r[3] for r in res])
    return Tc,kerma,flu,ev
Tc,kerma,flu,ev=comb(res)
pickle.dump((Tc,kerma,flu,ev,Ntot),open('/tmp/work/comb.pkl','wb'))
print(Tc['Lp']/Ntot/MC.VOL*MC.F0, Tc['Ls']/Ntot/MC.VOL*MC.F0, ev.shape)
