
s=open('/tmp/ana5.py').read().replace("/tmp/tallies.pkl","/tmp/tallies2.pkl")
s=s.replace("print('Phi_prim'","print('Kz full:',(Kz/Dmax).round(4)[:30]); print('Phi_prim'")
open('/tmp/ana5.py','w').write(s)
g={}
exec(s,g)
