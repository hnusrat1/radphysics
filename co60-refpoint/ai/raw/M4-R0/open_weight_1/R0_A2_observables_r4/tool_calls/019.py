import numpy as np
from scipy.interpolate import interp1d
ph=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
Eph=ph[:,0]; coh=ph[:,1]; inc=ph[:,2]; pe=ph[:,3]; pn=ph[:,4]; pel=ph[:,5]; tot=ph[:,6]
Elg=np.logspace(np.log10(0.01),np.log10(2.0),2500)
def interp_log(x,y,xq):
    ok=(y>0)&(x>0); fi=interp1d(np.log(x[ok]),np.log(y[ok]),kind='linear',bounds_error=False,fill_value='extrapolate')
    return np.exp(fi(np.log(xq)))
COH=interp_log(Eph,coh,Elg); INC_i=interp_log(Eph,inc,Elg); PE=interp_log(Eph,pe,Elg); TOT=interp_log(Eph,tot,Elg)
logElg=np.log(Elg)
def xs_lookup(arr,E): return np.exp(np.interp(np.log(np.maximum(E,2e-3)),logElg,np.log(arr)))
for E in [0.012,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.06,0.08,0.1]:
    mut=xs_lookup(TOT,np.array([E]))[0]
    s=xs_lookup(COH,np.array([E]))[0]+xs_lookup(INC_i,np.array([E]))[0]+xs_lookup(PE,np.array([E]))[0]
    print(f"E={E:.3f}: mut={mut:.4f} sum_no_pair={s:.4f} gap={1-s/mut:.5f}")
# check the pair spawning: the bug must be elsewhere; print pcoh+pinc+ppe for a low E in code terms