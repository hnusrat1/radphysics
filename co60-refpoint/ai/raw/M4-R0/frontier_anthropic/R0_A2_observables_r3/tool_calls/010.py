
print(np.round(o['Tsc']/o['Tsc'].sum(),4))
print(o['cnt'], o['etr'])
print(np.round(o['Tco']/o['Tco'].sum(),3))
print(o['birth_T'].size, o['birth_w'].sum())
print(np.round(o['Kdep'][:12]/1,6), np.round(o['Tdep'][:12],5))
