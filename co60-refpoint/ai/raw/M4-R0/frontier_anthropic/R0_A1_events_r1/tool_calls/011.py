
import json
events=json.load(open('answer.json'))['events']
def fmt(v): return 'null' if v is None else f'{v:.4g}'
lines=[]
for x in events:
    lines.append('["%s",%s,%s,%s,%s,%s,%s]'%(x['process'], fmt(x['e_in_MeV']), fmt(x['e_out_MeV']), fmt(x['cos_scatter']), fmt(x['t_e_MeV']), fmt(x['cos_electron']), fmt(x['t_positron_MeV'])))
open('compact.txt','w').write('\n'.join(lines))
print('\n'.join(lines[:100]))
