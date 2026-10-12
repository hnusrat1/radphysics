
import sys, pickle, json; sys.path.insert(0,'/tmp/work')
import numpy as np, xs, mell
from elec import EB25
tot=pickle.load(open('/tmp/work/tot.pkl','rb')); res=pickle.load(open('/tmp/work/eres1.pkl','rb'))
N=tot['N']; VC=np.pi*0.2; PHI=1/(0.0099751*110**2); norm=N*VC*PHI
fE=tot['flu_E']; fc=tot['flu_c']; enf=tot['en_fl']
EBF=np.arange(0,2.0001,0.01)
print('primary fl/phi0', fE[0].sum()/norm, 'scattered', fE[1].sum()/norm, 'total', fE.sum()/norm)
tot_fl=fE.sum(); scatfrac=fE[1].sum()/tot_fl
enfr=enf[1]/enf.sum()
meanE=enf.sum()/tot_fl
print('scat frac fl', scatfrac, 'energy frac', enfr, 'mean E', meanE, 'energy fl per phi0', enf/norm)
# 28 photon bins
edges28=np.arange(0,1.4001,0.05)
sc_spec=np.array([fE[1][int(round(edges28[i]/0.01)):int(round(edges28[i+1]/0.01))].sum() for i in range(28)])/norm
pr_spec=np.zeros(28)
print('scat spectrum sum', sc_spec.sum(), 'bins>1.4 fraction', 1-sc_spec.sum()/(fE[1].sum()/norm))
cedges=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0001])
ang_p=fc[0]/fc[0].sum(); ang_s=fc[1]/fc[1].sum()
print('ang prim',np.round(ang_p,4)); print('ang scat',np.round(ang_s,4))
# L2
keys=['coh','inc','pe','pr']
it=tot['intr']; et=tot['etr']
def frac(d,f=None):
    v={k:(d[k][f].sum() if f is not None else d[k].sum()) for k in keys}
    s=sum(v.values()); return {'compton':v['inc']/s,'photoelectric':v['pe']/s,'pair':v['pr']/s,'rayleigh':v['coh']/s}
print('L2 int', frac(it)); print('L2 energy', frac(et))
print('L2 prim', frac(it,0)); print('L2 scat', frac(it,1))
scshare=sum(it[k][1] for k in keys)/sum(it[k].sum() for k in keys)
print('scat share', scshare)
