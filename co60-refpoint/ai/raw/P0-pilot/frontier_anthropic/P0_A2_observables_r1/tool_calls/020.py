
S_phi0=0.009975072683099257*110.0**2; V=np.pi*0.2; Ntot=4000000
El=seg['E']; ll=seg['l']
ebin=np.digitize(El,EB)-1
spec_s=np.zeros(28)
ok=(ebin>=0)&(ebin<28)
np.add.at(spec_s,ebin[ok],ll[ok])
spec_s=spec_s*S_phi0/V/Ntot
over=ll[El>1.4].sum()*S_phi0/V/Ntot
print("Fl_sc check", spec_s.sum()+over, tot['Fl_sc'], "spec sum",spec_s.sum(),"over",over)
print("scat spectrum", np.round(spec_s,5))
EFs_check=(ll*El).sum()*S_phi0/V/Ntot
print("EFl check",EFs_check,tot['EFl'])
