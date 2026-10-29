import sys, os
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
s=open('mcbeam.py').read()
old="""        has_trk=trk>0
        if has_trk.any():
            photon_mc.debug['ntrk']+=int(has_trk.sum())
            photon_mc.debug['ntrkp']+=int((has_trk&(pra==1)).sum())
            photon_mc.debug['ntrks']+=int((has_trk&(pra==0)).sum())
            photon_mc.debug['sumtrk']+=float(trk[has_trk].sum()) """
new="""        has_trk=trk>0
        if has_trk.any():
            photon_mc.debug['ntrk']+=int(has_trk.sum())
            photon_mc.debug['ntrkp']+=int((has_trk&(pra==1)).sum())
            photon_mc.debug['ntrks']+=int((has_trk&(pra==0)).sum())
            photon_mc.debug['sumtrk']+=float(trk[has_trk].sum())
            if niter==1:
                photon_mc.debug['n1p']=photon_mc.debug.get('n1p',0)+int((has_trk&(pra==1)).sum())
                photon_mc.debug['n1ps']=photon_mc.debug.get('n1ps',0)+int((has_trk&(pra==1)&(s<5.1)).sum())
                photon_mc.debug['s1ps']=photon_mc.debug.get('s1ps',0.0)+float((trk*(has_trk&(pra==1)&(s<5.1))).sum()) """
if old not in s:
    print('pattern no')
else:
    s=s.replace(old,new)
    open('mcbeam.py','w').write(s)
    print('patched')