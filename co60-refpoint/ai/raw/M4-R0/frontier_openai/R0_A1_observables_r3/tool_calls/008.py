import numpy as np
w=np.array([.001,.002,.003,.004,.006,.009,.012,.016,.02,.025,.03,.035,.04,.043,.045,.045,.044,.042,.04,.038,.035,.03,.024,.017,.009,.003,.0005,.0001]); ws=w*.3/w.sum();print('scatter', [round(float(x),7) for x in ws],sum(ws))
# total initial arrays recompute from prior
me=.51099895; edges=np.array([.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1,1.05,1.1,1.15,1.2,1.4])
def eh(E,weight=1):
 mu=np.linspace(-1,1,12000); ep=E/(1+(E/me)*(1-mu));rat=ep/E;kn=rat**2*(rat+1/rat-(1-mu**2));return np.histogram(E-ep,edges,weights=kn*weight)[0]
ph=sum((eh(e) for e in (1.1732,1.3325)))/2; ph=ph/ph.sum()
centers=np.array([.025+i*.05 for i in range(28)])
sh=sum((eh(e,ww) for e,ww in zip(centers,w) if e>=.02));sh=sh/sh.sum();total=.7*ph+.3*sh
print('primary', [round(float(x),6) for x in ph],sum(ph))
print('total', [round(float(x),6) for x in total],sum(total))
print('L4prob', [round(float(x),6) for x in np.array([.06,.07,.075,.075,.075,.07,.065,.06,.055,.05,.045,.04,.035,.03,.025,.02,.015,.012,.01,.008,.006,.004,.003,.002,.001])/0.911])