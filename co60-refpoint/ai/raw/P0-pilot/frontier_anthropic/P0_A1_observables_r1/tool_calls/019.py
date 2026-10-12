
import pickle, json
R=pickle.load(open('/tmp/work/R.pkl','rb'))
out={k:R[k] for k in R if not k.startswith('_')}
def rnd(x):
    if isinstance(x,float): return float(f"{x:.5g}")
    if isinstance(x,list): return [rnd(v) for v in x]
    if isinstance(x,dict): return {k:rnd(v) for k,v in x.items()}
    return x
out=rnd(out)
out['method']=("Crude in-house Monte Carlo (no transport code available): analog photon transport in a 30 cm water cube, "
 "2 MeV point source at 100 cm SSD, uniform-solid-angle 10x10 cone, 4x10^7 histories. Compton via Klein-Nishina with exact kinematics; "
 "rough photoelectric, pair and coherent coefficients. Fluences, interaction rates and energy transfer scored by track length in the 1 cm x 0.2 cm cylinder, normalised to Phi0. "
 "Compton/photo/pair electron sources scored in the cylinder. Electron fluence from a CSDA-type recursion with Moller delta-ray production (infinite-medium, CPE assumed). "
 "Kerma from energy transfer, D = Kcol (CPE), radiative loss via 0.00927*T ratio. Spencer-Attix from restricted Bethe minus Moller delta energy above 10 keV. "
 "PDD from straight-ahead CSDA deposition of electrons born within 3 cm of the axis. Primary fluence is analytic exp(-mu*10) and agrees with MC. "
 "Scoring-volume spectra on the 2 MeV source fall outside the 0-1.4 MeV grid, so primary spectrum bins are zero. "
 "Bremsstrahlung and fluorescence are neglected; angular electron distribution is approximated.")
out['data_sources']=("Klein-Nishina cross section (analytic, Compton total = 0.0489 cm^2/g at 2 MeV, close to XCOM incoherent); "
 "Berger-Seltzer Bethe collision stopping power for water (I=75 eV, no density effect; checked against ESTAR at 0.1, 0.5 MeV); "
 "Moller delta cross section; air I=85.7 eV for Spencer-Attix denominator; radiative/collision ratio ~ Z/800*T with Z=7.4. "
 "Photoelectric, pair and coherent coefficients are rough power-law/approximate values (not XCOM); they contribute ~1-2% of interactions. "
 "No ESTAR/XCOM tables were used directly; no tabulated values were available offline.")
print(json.dumps(out,indent=1)[:12000])
