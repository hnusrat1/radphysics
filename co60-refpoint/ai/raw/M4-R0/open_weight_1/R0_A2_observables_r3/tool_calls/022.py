import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
me=0.51099895

# data (fast reload)
d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
_Em=d[:,0]; _incoh=d[:,2]; _pe=d[:,3]; _pair=d[:,4]+d[:,5]; _coh=d[:,1]
f_incoh=interp1d(_Em,_incoh,bounds_error=False,fill_value=(_incoh[0],_incoh[-1]))
f_pe=interp1d(_Em,_pe,bounds_error=False,fill_value=(_pe[0],_pe[-1]))
f_pair=interp1d(_Em,_pair,bounds_error=False,fill_value=(_pair[0],_pair[-1]))
f_coh=interp1d(_Em,_coh,bounds_error=False,fill_value=(_coh[0],_coh[-1]))
d2=np.loadtxt('/tmp/data/estar_water.txt',comments='#')
_Ee=d2[:,0]; _Y=d2[:,5]
f_Y=interp1d(_Ee,_Y,bounds_error=False,fill_value=(_Y[0],_Y[-1]))

PH_EDGES=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
                   0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
EL_EDGES=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
                   0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
phc=(PH_EDGES[:-1]+PH_EDGES[1:])/2
elc=(EL_EDGES[:-1]+EL_EDGES[1:])/2
E1=1.1732; E2=1.3325; Z=5.0

def kn_ratio(E,mu):
    a=E/me; r=1/(1+a*(1-mu)); return r*r*(r+1/r-(1-mu**2))

def compton_elec_pdf(E, E0_grid):
    """pdf dP/dE0 (per MeV) for the recoil electron of photon energy E, evaluated at E0_grid"""
    a=E/me
    mu=np.linspace(-1+1e-9,1-1e-9,4000)
    r=1/(1+a*(1-mu)); Ep=E*r; Ee=E-Ep
    dmu=mu[1]-mu[0]
    w=2*np.pi*kn_ratio(E,mu)*dmu
    dEedmu=np.abs(E*a*r**2)
    dPdE=w/dEedmu   # dP/dEe
    tot=np.sum(dPdE*(dEedmu*dmu))
    dPdE=dPdE/tot
    # interpolate onto E0_grid
    from scipy.interpolate import interp1d as ip
    # need pdf per MeV on grid: use histogram
    pdf=np.zeros(len(E0_grid))
    idx=np.clip(np.searchsorted(E0_grid,Ee,side='right')-1,0,len(E0_grid)-1)
    # accumulate weights dPdE*(bin width?) -- simpler: value of dP/dE at that Ee
    # map each mu point to nearest grid cell, contribution dP = dPdE*dEe, divide by grid cell width
    # use histogram of Ee with weights dP
    counts,bins=np.histogram(Ee,bins=np.append(E0_grid,E0_grid[-1]+0.01),weights=dPdE*dEedmu*dmu)
    widths=np.diff(np.append(E0_grid,E0_grid[-1]+0.01))
    pdf=counts/widths[:-1]
    return pdf, Ee, dPdE

def compton_cosphi_dist(E, ct_bins):
    """distribution of electron recoil cos(theta) for photon E, per ct bin (fraction)"""
    a=E/me
    mu=np.linspace(-1+1e-9,1-1e-9,4000)
    r=1/(1+a*(1-mu)); Ep=E*r
    sinmu=np.sqrt(np.maximum(1-mu**2,0))
    num=E-Ep*mu
    den=np.sqrt((E-Ep*mu)**2+(Ep*sinmu)**2)
    cosphi=num/den
    dmu=mu[1]-mu[0]
    w=2*np.pi*kn_ratio(E,mu)*dmu
    tot=w.sum()
    hist,bins=np.histogram(cosphi,bins=ct_bins,weights=w)
    return hist/tot

# ---- build photon fluence groups ----
r=np.load('mc_results.npz',allow_pickle=True)
Vcyl=np.pi*0.2
N=int(r['N']); i5=np.argmin(abs(r['depths']-5.0))
prim_tl=r['prim_tl'][i5]; scat_tl=r['scat_tl'][i5]
prim_phi=0.5*(105.0/105.0)**2*(np.exp(-0.0653626*5)+np.exp(-0.06122524*5))
k=prim_phi/(prim_tl/Vcyl/N)
spec_scat_per=r['spec_scat']/Vcyl/N*k

# primary lines fluence per Phi0
f1=0.5*np.exp(-0.0653626*5); f2=0.5*np.exp(-0.06122524*5)
groups_prim=[(E1,f1,'prim'),(E2,f2,'prim')]
groups_scat=[(phc[i],spec_scat_per[i],'scat') for i in range(28) if spec_scat_per[i]>0]

def interaction_rates(groups):
    Rc=Rpe=Rpair=Rray=0.0; TC=Tpe=Tpair=Tray=0.0
    for E,w,lab in groups:
        Rc+=w*f_incoh(E); Rpe+=w*f_pe(E); Rpair+=w*f_pair(E); Rray+=w*f_coh(E)
        # energy transfer
        a=E/me
        num,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu)*(E-E/(1+a*(1-mu))),-1,1)
        den,_=quad(lambda mu: 2*np.pi*kn_ratio(E,mu),-1,1)
        TeC=num/den if den>0 else 0
        TC+=w*f_incoh(E)*TeC
        Tpe+=w*f_pe(E)*E
        Tpair+=w*f_pair(E)*max(E-1.022,0)
        Tray+=0
    return dict(comp=Rc,pe=Rpe,pair=Rpair,rayleigh=Rray),dict(comp=TC,pe=Tpe,pair=Tpair,rayleigh=Tray)

def normfrac(dd):
    s=sum(dd.values())
    return {kk:vv/s for kk,vv in dd.items()}

Rall,Tall=interaction_rates(groups_prim+groups_scat)
Rp,Tp=interaction_rates(groups_prim)
Rs,Ts=interaction_rates(groups_scat)
print("interaction fractions all:",normfrac(Rall))
print("energy-transfer fractions all:",normfrac(Tall))
print("interaction fractions primary:",normfrac(Rp))
print("interaction fractions scattered:",normfrac(Rs))
print("scattered share of interactions:", sum(Rs.values())/(sum(Rs.values())+sum(Rp.values())))

# ---------------- L3 ----------------
def electron_source_bins(groups, EL_EDGES):
    # returns array over 25 bins of electron counts (per Phi0, arbitrary units) and mean energies and direction fractions
    nb=len(EL_EDGES)-1
    counts=np.zeros(nb); meanE=np.zeros(nb)
    dirfrac=np.zeros(10)
    ct=CT_EDGES if False else np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
    EG_fine=np.array([0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4])
    for E,w,lab in groups:
        # compton
        m_in=f_incoh(E)
        if m_in>0:
            pdf,Ee,dPdE = compton_elec_pdf(E, EG_fine)
            # distribute pdf (per MeV) into bins
            # Approx: probability in bin i = int pdf dE over bin
            # evaluate pdf values at bin centers by interpolation from EG_fine piecewise const -> use finer: integrate using Ee,dP pairs
            dP = dPdE*np.gradient(Ee)
            idx=np.clip(np.searchsorted(EL_EDGES,Ee,side='right')-1,0,nb-1)
            for i in range(nb):
                m=idx==i
                counts[i]+=w*m_in*dP[m].sum()
                meanE[i]+=w*m_in*(Ee[m]*dP[m]).sum() if m.any() else 0
            # direction
            cf=compton_cosphi_dist(E, ct)
            dirfrac+=w*m_in*cf
        # PE
        m_pe=f_pe(E)
        if m_pe>0:
            i=np.clip(np.searchsorted(EL_EDGES,E,side='right')-1,0,nb-1)
            counts[i]+=w*m_pe
            meanE[i]+=w*m_pe*E
            # PE direction: isotropic over ct bins: fraction = bin width/2
            binw=np.diff(ct)
            dirfrac+=w*m_pe*(binw/2)
        # pair
        m_p=f_pair(E)
        if m_p>0 and E>1.022:
            Te=(E-1.022)/2
            i=np.clip(np.searchsorted(EL_EDGES,Te,side='right')-1,0,nb-1)
            counts[i]+=2*w*m_p
            meanE[i]+=2*w*m_p*Te
            dirfrac[9]+=2*w*m_p
    return counts,meanE,dirfrac

counts_all,meanE_all,dir_all=electron_source_bins(groups_prim+groups_scat, EL_EDGES)
counts_p,meanE_p,dir_p=electron_source_bins(groups_prim, EL_EDGES)
L3_spec=counts_all/counts_all.sum()
L3_spec_p=counts_p/counts_p.sum()
L3_mean=meanE_all.sum()/counts_all.sum()
L3_mean_p=meanE_p.sum()/counts_p.sum()
print("\nL3 initial spectrum (all):",np.round(L3_spec,4))
print("L3 initial spectrum (primary parent):",np.round(L3_spec_p,4))
print("L3 mean init E all:",L3_mean," primary-parent:",L3_mean_p)
print("L3 direction all:",np.round(dir_all/dir_all.sum(),4))
print("L3 direction primary:",np.round(dir_p/dir_p.sum(),4))
np.save('L3_arrays.npy', np.array([counts_all,counts_p,meanE_all,dir_all],dtype=object), allow_pickle=True)