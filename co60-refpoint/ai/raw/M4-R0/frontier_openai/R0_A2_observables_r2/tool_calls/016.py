import numpy as np
# source histogram counts primary and scattered, sorted grid
Eedges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
mid=(Eedges[:-1]+Eedges[1:])/2
bp=np.array([272,356,322,342,301,306,288,314,321,308,303,287,288,355,324,383,417,456,636,395,293,363,156,0,0],float)
bs=np.array([2473,1313,652,430,331,242,195,164,146,113,107,75,84,71,48,47,37,23,20,13,8,3,0,0,0],float)
ref=171441.905976
rate=np.array([bp.sum(),bs.sum()])/ref
print('rates',rate, 'source binned electron count total',bp.sum(),bs.sum())
# ESTAR interpolate stopping powers log/log
estar=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
def stop(E,col=1): return np.exp(np.interp(np.log(np.clip(E,estar[0,0],estar[-1,0])),np.log(estar[:,0]),np.log(estar[:,col])))
# deterministic transport integrate for source electrons in each bin, starting energy midpoint
# source rate each energy bin = count/ref; electron spectrum: sum source electrons with E0 > energy * integral dE/S(E)
# get bins output
flu=np.zeros(25)
# energies the edges themselves; integrate over bin with quad
from scipy.integrate import quad
for j,(a,b) in enumerate(zip(Eedges[:-1],Eedges[1:])):
 # sources count above x midpoint approx fraction based within source bins (step source midpoint)
 xmid=(a+b)/2
 sources=(bp[mid>xmid].sum()+bs[mid>xmid].sum())/ref
 # integrate stopping from x bin edges, use energy divided by mass stopping power; g/cm2 at rho 1
 if a>=.01:
  flu[j]=sources*quad(lambda x:1/stop(x),max(a,.01),b,epsabs=1e-8)[0]
print('electron CSDA primary plus scat spectrum',flu,'sum',flu.sum())
print('avg energy',np.dot(flu,mid)/flu.sum())
# delta rays model 12%
