
import sys, os
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
s=open('mcbeam.py').read()
# count photons with primary track
s = s.replace("""        trk=track_in_cylinder(r0,r1c,R_cyl,zminl,zmaxl)
        has_trk=trk>0""",
"""        trk=track_in_cylinder(r0,r1c,R_cyl,zminl,zmaxl)
        if not hasattr(photon_mc,'debug'):
            photon_mc.debug = dict(ntrk=0, ntrkp=0, ntrks=0, sumtrk=0.0)
        has_trk=trk>0
        if has_trk.any():
            photon_mc.debug['ntrk']+=int(has_trk.sum())
            photon_mc.debug['ntrkp']+=int((has_trk&(pra==1)).sum())
            photon_mc.debug['ntrks']+=int((has_trk&(pra==0)).sum())
            photon_mc.debug['sumtrk']+=float(trk[has_trk].sum())""")
open('mcbeam.py','w').write(s)
print('ok')
