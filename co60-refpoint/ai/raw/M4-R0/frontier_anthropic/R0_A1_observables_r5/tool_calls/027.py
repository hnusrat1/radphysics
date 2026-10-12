
code = open('/tmp/work/an2.py').read()
code = code.replace("Stot=Scol+Srad\n", """Stot=Scol+Srad
# restricted stopping for continuous slowing (delta rays >10 keV treated discretely)
Mdel=np.zeros(Tg.size)
for i,T in enumerate(Tg):
    if T<0.02: continue
    e=Tg[(Tg>=0.01)&(Tg<=T/2)]
    if e.size>1:
        Mdel[i]=ne*np.trapezoid(e*ea.dsig_deps(T,e),e)
Sres=Stot-Mdel
print('check Sres/Stot at 0.1,0.5,1.0:', [float(Sres[np.argmin(abs(Tg-x))]/Stot[np.argmin(abs(Tg-x))]) for x in [0.1,0.5,1.0]])
Stot=Sres
""")
code = code.replace("print('Krad (MeV/cm, per Phi0)'", "print('Energy check: sum Phi*Scol(>=10keV)*dT, sum over all Phi*Stot ...');print('Krad (MeV/cm, per Phi0)'")
open('/tmp/work/an2.py','w').write(code)
o,e=runpy('/tmp/work/an2.py'); print(o[-5000:], e[-3000:])
