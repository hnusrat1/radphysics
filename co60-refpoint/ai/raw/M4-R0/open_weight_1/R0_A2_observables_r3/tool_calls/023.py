import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
me=0.51099895
d=np.loadtxt('/tmp/data/xcom_water.txt',comments='#')
_Em=d[:,0]; _incoh=d[:,2]; _pe=d[:,3]; _pair=d[:,4]+d[:,5]; _coh=d[:,1]
f_incoh=interp1d(_Em,_incoh,bounds_error=False,fill_value=(_incoh[0],_incoh[-1]))
f_pe=interp1d(_Em,_pe,bounds_error=False,fill_value=(_pe[0],_pe[-1]))
f_pair=interp1d(_Em,_pair,bounds_error=False,fill_value=(_pair[0],_pair[-1]))
f_coh=interp1d(_Em,_coh,bounds_error=False,fill_value=(_coh[0],_coh[-1]))
PH_EDGES=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,
                   0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])
EL_EDGES=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
                   0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
phc=(PH_EDGES[:-1]+PH_EDGES[1:])/2
ct=np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
E1=1.1732; E2=1.3325
def kn_ratio(E,mu):
    a=E/me; r=1/(1+a*(1-mu)); return r*r*(r+1/r-(1-mu**2))

def compton_elec(E):
    a=E/me
    mu=np.linspace(-1+1e-9,1-1e-9,4000)
    r=1/(1+a*(1-mu)); Ep=E*r; Ee=E-Ep
    dmu=mu[1]-mu[0]
    w=2*np.pi*kn_ratio(E,mu)*dmu
    dEe=np.abs(E*a*r**2)*dmu
    dP=w  # probability (normalized to tot)
    return Ee,dP

def compton_cosphi_dist(E):
    a=E/me
    mu=np.linspace(-1+1e-9,1-1e-9,4000)
    r=1/(1+a*(1-mu)); Ep=E*r
    sinmu=np.sqrt(np.maximum(1-mu**2,0))
    num=E-Ep*mu; den=np.sqrt((E-Ep*mu)**2+(Ep*sinmu)**2)
    cosphi=num/den
    w=2*np.pi*kn_ratio(E,mu)
    hist,bins=np.histogram(cosphi,bins=ct,weights=w)
    return hist/hist.sum()

r=np.load('mc_results.npz',allow_pickle=True)
Vcyl=np.pi*0.2; N=int(r['N']); i5=np.argmin(abs(r['depths']-5.0))
prim_phi=0.5*(np.exp(-0.0653626*5)+np.exp(-0.06122524*5))
k=prim_phi/(r['prim_tl'][i5]/Vcyl/N)
spec_scat_per=r['spec_scat']/Vcyl/N*k
f1=0.5*np.exp(-0.0653626*5); f2=0.5*np.exp(-0.06122524*5)
groups_prim=[(E1,f1),(E2,f2)]
groups_scat=[(phc[i],spec_scat_per[i]) for i in range(28) if spec_scat_per[i]>0]

def electron_source_bins(groups):
    nb=len(EL_EDGES)-1
    counts=np.zeros(nb); meanE=np.zeros(nb); dirfrac=np.zeros(10)
    for E,w in groups:
        m_in=f_incoh(E)
        if m_in>0:
            Ee,dP=compton_elec(E)
            tot=dP.sum()
            idx=np.clip(np.searchsorted(EL_EDGES,Ee,side='right')-1,0,nb-1)
            for i in range(nb):
                m=idx==i
                counts[i]+=w*m_in*dP[m].sum()
                meanE[i]+=w*m_in*(Ee[m]*dP[m]).sum()
            dirfrac+=w*m_in*compton_cosphi_dist(E)
        m_pe=f_pe(E)
        if m_pe>0:
            i=np.clip(np.searchsorted(EL_EDGES,E,side='right')-1,0,nb-1)
            counts[i]+=w*m_pe; meanE[i]+=w*m_pe*E
            binw=np.diff(ct); dirfrac+=w*m_pe*(binw/2)
        m_p=f_pair(E)
        if m_p>0 and E>1.022:
            Te=(E-1.022)/2
            i=np.clip(np.searchsorted(EL_EDGES,Te,side='right')-1,0,nb-1)
            counts[i]+=2*w*m_p; meanE[i]+=2*w*m_p*Te; dirfrac[9]+=2*w*m_p
    return counts,meanE,dirfrac

counts_all,meanE_all,dir_all=electron_source_bins(groups_prim+groups_scat)
counts_p,meanE_p,dir_p=electron_source_bins(groups_prim)
L3_spec=counts_all/counts_all.sum()
L3_spec_p=counts_p/counts_p.sum()
print("L3 mean init E all:",meanE_all.sum()/counts_all.sum()," primary-parent:",meanE_p.sum()/counts_p.sum())
print("L3 spec all:",np.round(L3_spec,4))
print("L3 spec primary:",np.round(L3_spec_p,4))
print("L3 dir all:",np.round(dir_all/dir_all.sum(),4))
print("L3 dir primary:",np.round(dir_p/dir_p.sum(),4))
np.save('L3_arrays.npy', np.array([counts_all,counts_p,meanE_all,dir_all,dir_p],dtype=object), allow_pickle=True)