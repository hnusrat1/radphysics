
s = open('/tmp/work/ana.py').read()
start = s.index("# ---------- L4 angular")
end = s.index("# ---------- L5 ----------")
newblock = r'''# ---------- L4 angular (Highland random walk along CSDA path) ----------
X0W = 36.08
def walk(Tb, cb, wb, nsteps=30):
    """returns histogram weights over cos bins of track-length weighted directions"""
    hist = np.zeros(10)
    T = Tb.copy(); c = cb.copy()
    sn = np.sqrt(np.maximum(1-c*c,0))
    for k in range(nsteps):
        m = T > 0.0101
        if not np.any(m): break
        Tn = np.maximum(T*0.8, 0.01)
        dR = np.maximum(Rc(T)-Rc(Tn), 0.0)
        dR = np.where(m, dR, 0.0)
        hist_add,_ = np.histogram(c[m], bins=CO, weights=(wb*dR)[m])
        hist += hist_add
        # Highland angle for step
        x = np.maximum(dR, 1e-9)/X0W
        pc = np.sqrt(T*(T+2*MEC2)); beta = pc/(T+MEC2)
        th0 = 13.6/(beta*pc)*np.sqrt(x)*(1+0.038*np.log(x))
        th0 = np.where(np.isfinite(th0), th0, 0)
        dx_ = rng.normal(0,1,T.size)*th0; dy_ = rng.normal(0,1,T.size)*th0
        dl = np.sqrt(dx_**2+dy_**2)
        ph = rng.random(T.size)*2*np.pi
        s_ = np.sqrt(np.maximum(1-c*c,0))
        cn = c*np.cos(dl) - s_*np.sin(dl)*np.cos(ph)
        c = np.clip(cn,-1,1)
        T = Tn
    return hist

# primary-origin e- births
selb = allsign==0
hist_prim = walk(allT[selb], allcos[selb], allw[selb])
# delta population
nd = 100000
pj = prod/prod.sum()
jj = rng.choice(nT, size=nd, p=pj)
eps_s = np.empty(nd); cos_d = np.empty(nd)
for k in range(nT):
    m_ = np.nonzero(jj==k)[0]
    if m_.size==0: continue
    Ep = Tnode[k]
    if Ep<=0.02: 
        eps_s[m_] = 0.01; cos_d[m_]=1.0; continue
    e = np.geomspace(0.01, Ep/2, 800)
    f = dsig_deltaE(Ep, e)*e
    cdf = np.concatenate([[0],np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(e))]); cdf/=cdf[-1]
    u = rng.random(m_.size)
    eps_s[m_] = np.interp(u, cdf, e)
    tau = Ep/MEC2
    cos_d[m_] = np.sqrt(np.clip(eps_s[m_]*(Ep+2*MEC2)/(Ep*(eps_s[m_]+2*MEC2)),0,1))
par_cos = rng.choice(allcos[selb], size=nd)
ph = rng.random(nd)*2*np.pi
sp = np.sqrt(np.maximum(1-par_cos**2,0)); sd = np.sqrt(np.maximum(1-cos_d**2,0))
cos_child = par_cos*cos_d + sp*sd*np.cos(ph)
Wd_ = prod.sum()/nd
hist_del = walk(eps_s, cos_child, np.full(nd, Wd_))
Htot = hist_prim+hist_del
L4_ang = Htot/Htot.sum()
print('L4 angular path-weights: prim', hist_prim.sum(), 'delta', hist_del.sum(), 'delta-fraction of track', hist_del.sum()/Htot.sum())

'''
s = s[:start] + newblock + s[end:]
open('/tmp/work/ana.py','w').write(s)
import subprocess
r=subprocess.run("cd /tmp/work && python ana.py",shell=True,capture_output=True,text=True)
import re
out = r.stdout
i = out.find('{')
print(out[:600])
print(r.stderr[-800:])
