import numpy as np
from scipy.interpolate import interp1d

xcom = np.loadtxt('/tmp/data/xcom_water.txt')
x_e = xcom[:,0]; x_inc = xcom[:,2]; x_tot = xcom[:,6]
estar = np.loadtxt('/tmp/data/estar_water.txt')
es_e = estar[:,0]; es_coll = estar[:,1]; es_rad = estar[:,2]; es_csda = estar[:,3]; es_ry = estar[:,4]

f_inc = interp1d(x_e, x_inc, kind='linear', bounds_error=False, fill_value=0.0)
f_tot = interp1d(x_e, x_tot, kind='linear', bounds_error=False, fill_value=0.0)
f_coll = interp1d(es_e, es_coll, kind='linear', bounds_error=False, fill_value=0.0)
f_rad = interp1d(es_e, es_rad, kind='linear', bounds_error=False, fill_value=0.0)
f_csda = interp1d(es_e, es_csda, kind='linear', bounds_error=False, fill_value=0.0)

mec2=0.511; rho=1.0; E0=2.0; SSD=100.0
mu_tot_0 = float(f_tot(E0)); mu_inc_0 = float(f_inc(E0))
L1_pri = np.exp(-mu_tot_0*10.0)
print(f"L1_pri = {L1_pri:.6f}")

# First scatter integration
nz=60; nr=30
z_a = np.linspace(0.25, 29.75, nz)
r_a = np.linspace(0.15, 14.85, nr)
dz=30.0/nz; dr=15.0/nr

ph_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
cos_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
NA=6.022e23; Ne_g=NA*10/18.015; re=2.818e-13

L1_scat=0.0; sc_spec=np.zeros(28); sc_ang=np.zeros(10)

for iz in range(nz):
    z = z_a[iz]
    hw = 5.0*(z+SSD)/SSD
    for ir in range(nr):
        r = r_a[ir]
        if r > hw: continue
        dV = 2*np.pi*r*dr*dz
        d_sv = np.sqrt(r**2 + (z+SSD)**2)
        ca = (z+SSD)/d_sv
        pw = z/ca
        pf = (110.0/(z+SSD))**2 * np.exp(-mu_tot_0*pw)
        dNs = pf * mu_inc_0 * dV
        
        dzs = 10.0 - z
        ds = np.sqrt(r**2 + dzs**2)
        if ds < 0.01: continue
        
        cts = (r*(-1) + 0 + (z+SSD)*dzs) / (d_sv*ds)
        cts = np.clip(cts, -1, 1)
        ts = np.arccos(cts)
        Ep = E0/(1+(E0/mec2)*(1-cts))
        
        r_kn = Ep/E0
        s2 = np.sin(ts)**2
        dsdO = (re**2/2)*r_kn**2*(1/r_kn+r_kn-s2)
        
        cb = dzs/ds
        Om = np.pi*1.0**2*abs(cb)/ds**2
        sig_inc = mu_inc_0/Ne_g
        p_sc = (dsdO/sig_inc)*Om/(4*np.pi)
        
        mut_sc = float(f_tot(Ep))
        att_sc = np.exp(-mut_sc*ds)
        
        proj = np.pi*1.0**2*abs(cb)
        if proj > 1e-10:
            dPh = dNs * p_sc * att_sc / proj
            L1_scat += dPh
            if 0 < Ep <= 1.4:
                bi = np.searchsorted(ph_bins, Ep) - 1
                if 0 <= bi < 28: sc_spec[bi] += dPh
            cth = dzs/ds
            bi = np.searchsorted(cos_bins[1:], cth, side='right')
            bi = min(bi, 9)
            sc_ang[bi] += dPh

print(f"\nL1_scattered_fluence = {L1_scat:.6f}")
print(f"L1_total_fluence = {L1_pri+L1_scat:.6f}")
sf = L1_scat/(L1_pri+L1_scat)
print(f"Scattered fraction fluence = {sf:.6f}")
print(f"\nScattered spectrum sum = {np.sum(sc_spec):.6f}")
print(f"Scattered angular sum = {np.sum(sc_ang):.6f}")