
ev200 = out['events']
def fmt(x, nd=4):
    return 'null' if x is None else (f"{x:.{nd}g}" if abs(x)>=1e-3 or x==0 else f"{x:.3e}")
lines=[]
for e in ev200:
    lines.append('{"process": "%s", "e_in_MeV": %s, "e_out_MeV": %s, "cos_scatter": %s, "t_e_MeV": %s, "cos_electron": %s, "t_positron_MeV": %s}' % (
        e['process'], fmt(e['e_in_MeV'],6), fmt(e['e_out_MeV'],6), fmt(e['cos_scatter'],5), fmt(e['t_e_MeV'],6), fmt(e['cos_electron'],5), fmt(e['t_positron_MeV'],6)))
print("\n".join(lines[:70]))
