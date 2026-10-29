from collections import Counter
print(Counter(e['process'] for e in reservoir))
comps=[e for e in reservoir if e['process']=='compton']
print('Ein quantiles',np.quantile([e['e_in_MeV'] for e in comps],[0,.1,.25,.5,.75,.9,1]))
print('Eout quantiles',np.quantile([e['e_out_MeV'] for e in comps],[0,.1,.25,.5,.75,.9,1]))
print('primary line counts',Counter(round(e['e_in_MeV'],4) for e in comps if e['e_in_MeV']>1.1))
print('sample first 10',reservoir[:10])