import numpy as np
from scipy.integrate import quad
estar=np.loadtxt('/tmp/data/estar_water.txt')
energy=estar[:,0]; scoll=estar[:,1]; range_csda=estar[:,4]
# interpolate 1/S (log-log) and range
invS=lambda E: np.exp(np.interp(np.log(np.maximum(E,energy[0])),np.log(energy),np.log(1/scoll)))
# choose conditional distributions hp, hs as arrays computed previous unavailable execute rebuild simple values hard to persist; recompute read from literals? use approximate ht/hp captured
hp=np.array([0.034820198700304575,0.0429026722200118,0.04229075566542619,0.04169929548309062,0.04092957386558756,0.040462160787453245,0.0401146581124793,0.03958788996330752,0.039531153549685616,0.03937019915994527,0.039654027089374135,0.03998053141822054,0.040893479266078496,0.04231225523564851,0.04464138149845266,0.04821618869106213,0.05398549000087125,0.06331139881160865,0.07952412310589158,0.04665598682106448,0.03427399924667309,0.044636868440795646,0.02020571286696723,0,0])
hs=np.array([0.3612733257666176,0.19807011072317413,0.11483002316775318,0.07443496098520222,0.054037955158658134,0.04067208042704748,0.03144456660497912,0.023983122849410277,0.01894454194925498,0.015231609770840461,0.012801661815756845,0.010584472497638822,0.008684631782737194,0.007423750804036568,0.006234879592999909,0.005072023902823645,0.004157841196487945,0.00352547592279937,0.0029005565534968914,0.0022682266218903733,0.0015766848331703355,0.0010841301699185655,0.0006458302149222804,0.00011753668838369753,0])
# using midpoint approx for source electron start energies; aggregate in 0.01-bin track energy segments
bins=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
mid=(bins[:-1]+bins[1:])/2
# use electron event contributions: rates primary and scattered approximate conditional distributions. Compton interaction rates
rp=.04595; rs=.01465
track=np.zeros(len(bins)-1)
range_avg=0
for dist,rate in [(hp,rp),(hs,rs)]:
 for prob,E0 in zip(dist,mid):
  if prob==0:continue
  E0=max(E0,.01)
  for j,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
   top=min(hi,E0)
   if top>lo:
    track[j]+=rate*prob*quad(invS,lo,top)[0]
  range_avg+=rate*prob*quad(invS,.01,E0)[0]
print('range weighted track fluence before delta',track.sum(),'mean range events',range_avg, 'source',rp+rs)
print('conditional mean range',range_avg/(rp+rs),'track spectrum frac',track/track.sum())
print('ESTAR range max',range_csda[-1])