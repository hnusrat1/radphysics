import numpy as np
L3=np.load('L3.npz'); L4=np.load('L4b.npz')
Egrid=L4['Egrid']
h_all=L3['h_all']/float(L3['wtot'])
wE=float(L3['wE']); wtot=float(L3['wtot']); Emean=wE/wtot
edges_e=np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])
def p_of_E(E):
    i=int(np.searchsorted(edges_e,E,side='right')-1); i=max(0,min(24,i))
    return h_all[i]/(edges_e[i+1]-edges_e[i])
p=np.array([p_of_E(E) for E in Egrid])
Kcol_Phi=5.352357476400485e-12
q=Kcol_Phi*6.242e9*p/Emean
q=q*(Kcol_Phi*6.242e9)/np.trapezoid(q*Egrid,Egrid)
Ew=np.array([0.01,0.015,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.5])
Sw=np.array([22.36,16.10,12.98,9.00,5.93,4.72,4.10,3.35,3.05,2.63,2.39,2.23,2.12,2.04,1.98,1.93,1.89,1.86,1.84,1.83,1.84],float)
def Swater(E): return float(np.exp(np.interp(np.log(E),np.log(Ew),np.log(Sw))))
def g_rad(E0):
    eg=Egrid[Egrid<=E0]
    if len(eg)<2: return 0.0
    ratio=np.array([(0.006*e**1.7)/max(Swater(e),1e-9) for e in eg])
    return np.trapezoid(ratio,eg)/E0
gr=np.array([g_rad(E) for E in Egrid])
print("max gr",gr.max(),"mean gr",gr.mean())
we=q*Egrid
idx=np.argsort(-gr)[:10]
print("top gr at E:",Egrid[idx],gr[idx])
# weighted average contribution
prod=we*gr
i=np.argmax(prod)
print("max prod at E:",Egrid[i], "prod:",prod[i])
print("sum prod*dE:",np.trapezoid(prod,Egrid))
print("integral qE:",np.trapezoid(we,Egrid))
# where is prod big? histogram
for lo,hi in [(0.01,0.1),(0.1,0.3),(0.3,0.6),(0.6,1.0),(1.0,1.4)]:
    m=(Egrid>=lo)&(Egrid<hi)
    print(lo,hi,"prod sum:",np.trapezoid(prod[m],Egrid[m]),"gr mean:",gr[m].mean())
