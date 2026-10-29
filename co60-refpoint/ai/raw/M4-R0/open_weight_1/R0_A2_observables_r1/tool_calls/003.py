import numpy as np, math

# Load XCOM
d = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
E_x = d[:,0]; mu_co = d[:,1]; mu_in = d[:,2]; mu_pe = d[:,3]; mu_pn = d[:,4]; mu_pefield = d[:,5]; mu_tot = d[:,6]
# ESTAR
s = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
Ee = s[:,0]; Scol = s[:,1]; Srad = s[:,2]; Stot = s[:,3]; R_csda = s[:,4]; Y = s[:,5]; debye = s[:,6]

def interp(x, xp, yp):
    return np.interp(x, xp, yp)

def mu_total(E):  # cm^2/g
    return interp(E, E_x, mu_tot)
def mu_coh(E): return interp(E,E_x,mu_co)
def mu_inc(E): return interp(E,E_x,mu_in)
def mu_pe_(E): return interp(E,E_x,mu_pe)
def mu_pair(E): return interp(E,E_x,mu_pn)+interp(E,E_x,mu_pefield)

print('mu/rho at 1.1732:', mu_total(1.1732), 'coh',mu_coh(1.1732),'inc',mu_inc(1.1732),'pe',mu_pe_(1.1732),'pair',mu_pair(1.1732))
print('mu/rho at 1.3325:', mu_total(1.3325), 'coh',mu_coh(1.3325),'inc',mu_inc(1.3325),'pe',mu_pe_(1.3325),'pair',mu_pair(1.3325))
for E in (1.1732,1.3325):
    print(E, 'exp(-mu*5) =', math.exp(-mu_total(E)*5))

# Klein-Nishina: differential cross section dsigma/dcos, and energy transfer
mc2 = 0.5109989461
def KN_f(alpha, cos):
    x = 1+alpha*(1-cos)
    r = 1.0/x
    return r*r*(r + x - (1-cos*cos))
# mean fractional energy transfer Tbar/E = 1 - <E'>/E
def compton_mean_transfer(E, n=200000, seed=1):
    rng=np.random.default_rng(seed)
    alpha = E/mc2
    cos = rng.uniform(-1,1,n)
    f = KN_f(alpha,cos)
    # rejection sample with envelope 2
    acc = rng.random(n) < f/2.0
    cs = cos[acc]
    x = 1+alpha*(1-cs)
    Ep = E/x
    T = E-Ep
    return T.mean(), T.std(), len(cs)/n
for E in (1.1732,1.3325):
    tm,ts,acc=compton_mean_transfer(E)
    print(E,'mean T_e', tm, 'Tbar/E', tm/E, 'acceptance', acc)

# mu_tr/rho at line energies (approx, neglecting pair small and pe Eb)
def mu_tr_rho(E):
    alpha=E/mc2
    # sample mean transfer via integration
    # integrate KN dsigma/dcos numerically
    cos=np.linspace(-1,1,4001)
    f=KN_f(alpha,cos)
    x=1+alpha*(1-cos)
    Ep=E/x
    T=E-Ep
    # integral = int f dcos ; weighted mean T
    w=f*(1-cos*cos)  # careful: KN formula already (1+cos^2)?? no: r^2(r+1/r - sin^2) with r=E'/E
    # dsigma/dcos ∝ r^2 (r + 1/r - sin^2): f above. average T = int T f dcos / int f dcos
    meanT = np.trapezoid(T*f, cos)/np.trapezoid(f, cos)
    mu_t = mu_inc(E)*meanT/E + mu_pe_(E)*(E-0.0005)/E + mu_pair(E)*(E-1.022)/E
    return mu_t, meanT
for E in (1.1732,1.3325):
    mt,meanT = mu_tr_rho(E)
    print(E,'mu_tr/rho',mt,'meanT',meanT)
    
print('V_cyl =', math.pi*1.0**2*0.2)
print('phi0 per photon =', 1/10.5**2)