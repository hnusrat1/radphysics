
L=open('lines.txt').read().split('\n')
print(len(L))
short=[]
import json
for l in L:
    o=json.loads(l)
    short.append([o['process'][:2] if o['process']!='photoelectric' else 'pe', o['e_in_MeV'], o['e_out_MeV'], o['cos_scatter'], o['t_e_MeV'], o['cos_electron']])
print(json.dumps(short[:100],separators=(',',':')))
