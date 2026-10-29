
import pickle, numpy as np, sys
sys.path.insert(0,'.')
import mc2
allres=pickle.load(open('allres.pkl','rb'))
depths=[0.25,0.5,0.75,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0]
Ntot=12000000; V=mc2.Vcyl; P0=mc2.Phi0_hist; c5=depths.index(5.0)
Hfine=sum(r['Hfine'] for r in allres); Hcoarse=sum(r['Hcoarse'] for r in allres)
Hcos=sum(r['Hcos'] for r in allres); Esum=sum(r['Esum'] for r in allres); Wsum=sum(r['Wsum'] for r in allres)
fine=allres[0]['fine']; eedges=allres[0]['eedges']
re=2.8179403e-13; mc2_=0.51099895; NA=6.02214076e23
ne_w=0.55509*NA; ne_a=0.49919*NA
Kw=2*np.pi*re**2*mc2_*ne_w
def Scol(T, I_eV, ne_per_g):
    tau=np.asarray(T)/mc2_; g=tau+1; b2=1-1/g**2
    Kg=2*np.pi*re**2*mc2_*ne_per_g
    Im=I_eV*1e-6/mc2_
    F=1-b2+(tau**2/8-(2*tau+1)*np.log(2))/g**2
    return Kg/b2*(np.log(tau**2*(tau+2)/(2*Im**2))+F)
def Yrad(T): return np.minimum(0.0069*T,0.5)
Ec=np.sqrt(fine[:-1]*fine[1:]); dE=np.diff(fine)
Scw=Scol(Ec,75.0,ne_w); Stw=Scw/(1-Yrad(Ec)); Srw=Stw-Scw
Qph=(Hfine[c5,0]+Hfine[c5,1])/(Ntot*V)/dE
print("sum Qph*E*dE /P0 *1.602e-10 =", np.sum(Qph*Ec*dE)/P0*1.602e-10)
Kc_tr=sum(allres[0]['tal'] and 0 for _ in [0])
# recompute track Kcol
tal={k:sum(r['tal'][k] for r in allres) for k in allres[0]['tal']}
Kc_track=sum(tal[f'K{p}_{s}'][c5] for p in ['C','PE','PP'] for s in ['p','s'])/(Ntot*V*P0)*1.602e-10
print("track Kcol", Kc_track)
