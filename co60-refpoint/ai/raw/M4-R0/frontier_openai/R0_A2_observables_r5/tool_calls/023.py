Scoll=np.exp(interp1d(np.log(estar[:,0]),np.log(estar[:,1]),fill_value='extrapolate')(np.log(mid)))
print((Fl*Scoll).sum(), 'ratio to kerma')