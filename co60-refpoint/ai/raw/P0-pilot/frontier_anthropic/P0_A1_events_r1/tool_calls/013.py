
import json,collections,numpy as np
ev=[]
for s in range(2,6): ev+=json.load(open(f'events_{s}.json'))
print(len(ev), collections.Counter(e['process'] for e in ev))
prim=[e for e in ev if e['e_in_MeV']==2.0]
print('primary-energy interactions', len(prim), collections.Counter(e['process'] for e in prim))
c=[e for e in ev if e['process']=='compton']
ein=np.array([e['e_in_MeV'] for e in c]); print('compton e_in quantiles', np.percentile(ein,[5,25,50,75,95]))
print('frac e_in==2', np.mean(ein==2.0))
print('pair', [ (e['e_in_MeV'],e['t_e_MeV'],e['t_positron_MeV']) for e in ev if e['process']=='pair'][:5])
ray=[e for e in ev if e['process']=='rayleigh']; print('rayleigh e_in', collections.Counter(round(e['e_in_MeV'],3) for e in ray).most_common(5))
pe=[e for e in ev if e['process']=='photoelectric']; print('pe', collections.Counter(round(e['e_in_MeV'],3) for e in pe).most_common(5))
