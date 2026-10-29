
import pickle, numpy as np, sys
sys.path.insert(0,'.')
import mc2
allres=pickle.load(open('allres.pkl','rb'))
depths=[0.25,0.5,0.75,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0]
NH=sum(12000000 for _ in [0])//1  # total histories
Ntot=12000000
def comb(key):
    return sum(r['tal'][key] for r in allres)
tal={k:comb(k) for k in allres[0]['tal']}
Hfine=sum(r['Hfine'] for r in allres); Hcoarse=sum(r['Hcoarse'] for r in allres)
Hcos=sum(r['Hcos'] for r in allres); Esum=sum(r['Esum'] for r in allres); Wsum=sum(r['Wsum'] for r in allres)
fine=allres[0]['fine']; eedges=allres[0]['eedges']
V=mc2.Vcyl; P0=mc2.Phi0_hist
c5=depths.index(5.0); cd=[depths.index(0.5)]
norm=Ntot*V*P0
# L1
Fp=tal['Lp'][c5]/norm; Fs=tal['Ls'][c5]/norm
print("Fp",Fp,"Fs",Fs,"Ftot",Fp+Fs, "scat frac", Fs/(Fp+Fs))
print("energy scat frac", tal['LE_s'][c5]/(tal['LE_s'][c5]+tal['LE_p'][c5]))
print("mean E", (tal['LE_s'][c5]+tal['LE_p'][c5])/(tal['Ls'][c5]+tal['Lp'][c5]))
Eedges=np.arange(0,1.4001,0.05)
sp=tal['Lp_E'][c5]/norm; ss=tal['Ls_E'][c5]/norm
print("primary spectrum sum",sp.sum(), "scatt", ss.sum())
print("prim angular", tal['Lp_c'][c5]/tal['Lp_c'][c5].sum())
print("scat angular", np.round(tal['Ls_c'][c5]/tal['Ls_c'][c5].sum(),4))
# interactions
procs=['C','PE','PP','R']
Ip={p:tal[f'I{p}_p'][c5] for p in procs}; Is={p:tal[f'I{p}_s'][c5] for p in procs}
Kp={p:tal[f'K{p}_p'][c5] for p in procs}; Ks={p:tal[f'K{p}_s'][c5] for p in procs}
Itot={p:Ip[p]+Is[p] for p in procs}
print("interaction fractions", {p:Itot[p]/sum(Itot.values()) for p in procs})
print("energy transfer fractions", {p:(Kp[p]+Ks[p])/sum(Kp[q]+Ks[q] for q in procs) for p in procs})
print("primary int fractions", {p:Ip[p]/sum(Ip.values()) for p in procs})
print("scattered int fractions", {p:Is[p]/sum(Is.values()) for p in procs})
print("scattered share", sum(Is.values())/sum(Itot.values()))
Kcol_tot=(sum(Kp.values())+sum(Ks.values()))/norm*1.602e-10
print("Kcol/Phi0 Gy cm2", Kcol_tot)
print("Kcol by depth:")
for c,dz in enumerate(depths):
    kk=(sum(tal[f'K{p}_p'][c] for p in procs)+sum(tal[f'K{p}_s'][c] for p in procs))/(Ntot*V*P0)*1.602e-10
    print(dz, kk, tal['Lp'][c]/(Ntot*V*P0)+tal['Ls'][c]/(Ntot*V*P0))
