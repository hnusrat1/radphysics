from scipy.integrate import quad
for E in [1.1732,1.3325]:
 Tm=2*E*E/(m+2*E)
 tot=quad(lambda t:kn_pdf(t,E),0,Tm)[0]
 en=quad(lambda t:t*kn_pdf(t,E),0,Tm)[0]/tot
 print(Tm,en)
