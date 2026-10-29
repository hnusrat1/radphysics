delta=np.diff(ee)/np.sqrt(mids);delta[mids>0.55]=0;delta=delta/delta.sum()
base=raw/raw.sum(); allsp=.8*base+.2*delta
print(delta.tolist());print(allsp.tolist(), 'mean',allsp@mids)