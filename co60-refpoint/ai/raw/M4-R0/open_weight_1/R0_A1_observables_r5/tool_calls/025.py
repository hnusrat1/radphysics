
import sys, os
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
s=open('mcbeam.py').read()
s=s.replace("""        trk=track_in_cylinder(r0,r1c,R_cyl,zminl,zmaxl)
        if not hasattr(photon_mc,'debug') or photon_mc.debug is None:
            photon_mc.debug = dict(ntrk=0, ntrkp=0, ntrks=0, sumtrk=0.0)""",
"""        trk=track_in_cylinder(r0,r1c,R_cyl,zminl,zmaxl)
        if not hasattr(photon_mc,'debug') or photon_mc.debug is None:
            photon_mc.debug = dict(ntrk=0, ntrkp=0, ntrks=0, sumtrk=0.0, sfrac=0, sfracN=0, ssum=0.0)
        if niter==1:
            photon_mc.debug['sfrac']+=int((s<5.1).sum()); photon_mc.debug['sfracN']+=n_alive
            photon_mc.debug['ssum']+=float(s.sum())""")
open('mcbeam.py','w').write(s)
print('ok')
