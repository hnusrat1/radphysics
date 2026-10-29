
import numpy as np

mec2=0.5110
N_e_per_g = 6.02214076e23*10/18.01528
E_tab = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0])
mu_tab = np.array([4.951,1.374,0.7031,0.4470,0.2269,0.2059,0.1837,0.1706,0.1505,0.1370,0.1186,0.1061,0.0969,0.0896,0.0786,0.0707,0.0636,0.0576,0.0494])
coh_tab = np.array([0.0,0.0,0.0,0.03,0.0205,0.0148,0.0102,0.0084,0.0047,0.0030,0.0014,0.0009,0.0006,0.00045,0.00030,0.00024,0.00019,0.00016,0.00010])
def interp(x,xe,ye): return np.interp(x,xe,ye)
def sig_KN(E):
    al=E/mec2; a2=al*al
    f=(1+al)/a2*(2*(1+al)/(1+2*al)-np.log(1+2*al)/al)+np.log(1+2*al)/(2*al)-(1+3*al)/(1+2*al)**2
    return 0.4987*f
def mu_C(E): return sig_KN(E)*1e-24*N_e_per_g
def mu_coh(E): return interp(E,E_tab,coh_tab)
def mu_pair(E): return np.where(E>1.022, 0.00052*((E-1.022)/0.978)**2.0, 0.0)
def mu_pe(E): return np.maximum(0, mu_tot(E)-mu_C(E)-mu_coh(E)-mu_pair(E))
def mu_tot(E): return interp(E,E_tab,mu_tab)

def kn_diff(cosT,E):
    al=E/mec2; x=1+al*(1-cosT)
    return (1.0/x)**2*(0.5*(1+cosT*cosT)+(al*al*(1-cosT)**2)/(2*x))

def compton_electrons(E, n, rng):
    # sample KN photon scatter angles, return electron kinetic energies and cosphi
    x = rng.uniform(-1,1,size=n)
    acc = rng.uniform(0,1,size=n) < kn_diff(x,E)
    while not acc.all():
        rem=~acc
        nx=rng.uniform(-1,1,size=rem.sum())
        nacc = rng.uniform(0,1,size=rem.sum())<kn_diff(nx,E)
        x[rem]=nx; acc[rem]=nacc
    al=E/mec2
    Ep = E/(1+al*(1-x))
    Ee = E-Ep
    pe = np.sqrt(np.maximum(Ee*(Ee+2*mec2),0))
    cosphi = np.clip((E-Ep*x)/pe,-1,1)
    return Ee, cosphi

ph_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
ph_c = (ph_bins[:-1]+ph_bins[1:])/2
el_bins = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
el_c = (el_bins[:-1]+el_bins[1:])/2
ct_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

prim_spec = np.load('prim_spec.npy')
scat_spec = np.load('scat_spec.npy')
tot_spec = prim_spec+scat_spec

rng = np.random.default_rng(123)
def initial_spectra(spec):
    # returns histogram (25), mean energy, angular (10)  for electrons+positrons set in motion
    hist = np.zeros(25); ang = np.zeros(10); wsum = 0.0; esum=0.0
    for i,Ec in enumerate(ph_c):
        wf = spec[i]
        if wf<=0: continue
        nC = wf*mu_C(Ec)
        if nC>0:
            n_samp = 20000
            Ee, cp = compton_electrons(Ec, n_samp, rng)
            w = nC/n_samp
            h = np.histogram(Ee, bins=el_bins, weights=np.full(n_samp,w))[0]
            hist += h
            a = np.histogram(cp, bins=ct_bins, weights=np.full(n_samp,w))[0]
            ang += a
            esum += w*Ee.sum()
            wsum += w*n_samp
        nPE = wf*mu_pe(Ec)
        if nPE>0:
            T = Ec-0.0005
            if T>=0.01:
                # bin
                bi = np.searchsorted(el_bins,T,side='right')-1
                if bi<0: bi=0
                if bi>=25: bi=24
                hist[bi]+=nPE
                ang[9]+=nPE
                esum += nPE*T
                wsum += nPE
        nPA = wf*mu_pair(Ec)
        if nPA>0:
            T = Ec-1.022
            if T>0:
                for _ in range(2):  # electron and positron
                    f = rng.uniform(0,1)
                    Te = f*T
                    if Te>=0.01:
                        bi = np.searchsorted(el_bins,Te,side='right')-1
                        if bi<0: bi=0
                        if bi>=25: bi=24
                        hist[bi]+=nPA
                        ang[9]+=nPA
                        esum += nPA*Te
                        wsum += nPA
    return hist, esum/wsum, ang, wsum

hist_tot, mean_tot, ang_tot, w_tot = initial_spectra(tot_spec)
hist_prim, mean_prim, ang_prim, w_prim = initial_spectra(prim_spec)

print("L3 initial spectrum (all):", np.round(hist_tot/hist_tot.sum(),5))
print("sum:", hist_tot.sum())
print("L3 mean initial energy (all):", mean_tot)
print("L3 initial spectrum (parent primary):", np.round(hist_prim/hist_prim.sum(),5))
print("L3 mean initial energy (parent primary):", mean_prim)
print("L3 initial direction (all):", np.round(ang_tot/ang_tot.sum(),5))
print("L3 initial direction (should sum 1):", (ang_tot/ang_tot.sum()).sum())
np.save('L3_data.npy', {'hist_tot':hist_tot,'mean_tot':mean_tot,'ang_tot':ang_tot,'hist_prim':hist_prim,'mean_prim':mean_prim,'ang_prim':ang_prim,'w_tot':w_tot,'w_prim':w_prim})
