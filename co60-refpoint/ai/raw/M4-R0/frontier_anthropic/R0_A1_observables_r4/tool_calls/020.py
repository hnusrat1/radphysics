
Ered=np.sum(Fl*Sres*dE)*conv
print('restricted check  sum Phi*Sres =',Ered,' vs photon kerma-based', K_per_phi0, 'ratio', Ered/K_per_phi0)
print('sum Phi*Scol ratio',Ecol/K_per_phi0)
# energy in parent (primary) sources only, Q_src contributions
