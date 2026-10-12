
import json
ev200 = json.load(open('answer.json'))['events']
def fmt(x, nd=6):
    return 'null' if x is None else f"{x:.{nd}g}"
lines=[]
for e in ev200:
    lines.append('{"process": "%s", "e_in_MeV": %s, "e_out_MeV": %s, "cos_scatter": %s, "t_e_MeV": %s, "cos_electron": %s, "t_positron_MeV": %s}' % (
        e['process'], fmt(e['e_in_MeV']), fmt(e['e_out_MeV']), fmt(e['cos_scatter'],5), fmt(e['t_e_MeV']), fmt(e['cos_electron'],5), fmt(e['t_positron_MeV'])))
open('lines.txt','w').write("\n".join(lines))
print("\n".join(lines[:65]))
