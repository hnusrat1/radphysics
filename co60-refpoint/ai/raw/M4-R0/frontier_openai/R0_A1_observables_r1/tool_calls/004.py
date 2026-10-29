prim=.737; scat=.16; mp=1.25285; ms=.698379; muenp=.0295;muenS=.032
Kcol=1.602176634e-10*(prim*mp*muenp+scat*ms*muenS)
print(Kcol, 'phi energy', prim*mp+scat*ms)
# weighted kerma total using mutr = muen/(1-g) approx g=.025 avg primary, .03 scat
K=1.602176634e-10*(prim*mp*muenp/(1-.025)+scat*ms*muenS/(1-.03))
print(K,(K-Kcol)/K, Kcol/K)
