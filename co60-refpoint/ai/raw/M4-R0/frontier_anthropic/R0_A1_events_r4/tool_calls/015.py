
import numpy as np, json
d=np.load('sample_99.npz')
names=['compton','rayleigh','photoelectric','pair']
lines=[]
for i in range(len(d['proc'])):
    p=names[int(d['proc'][i])]
    eo=float(d['Eout'][i]); cs=float(d['cosS'][i]); te=float(d['te'][i]); ce=float(d['cose'][i]); tp=float(d['tpos'][i]); e_in=float(d['Ein'][i])
    ev={'process':p,'e_in_MeV':round(e_in,6),'e_out_MeV':None if np.isnan(eo) else round(eo,6),
        'cos_scatter':None if np.isnan(cs) else round(cs,5),'t_e_MeV':round(te,6),
        'cos_electron':None if np.isnan(ce) else round(ce,5),'t_positron_MeV':None if np.isnan(tp) else round(tp,6)}
    if p=='rayleigh':
        ev['e_out_MeV']=round(e_in,6); ev['t_e_MeV']=0.0; ev['cos_electron']=None
    lines.append(json.dumps(ev,separators=(',',':')))
print(',\n'.join(lines[25:60]))
