
def frac(d,keys):
    tot=sum(d.values()); return {k: sum(d[(p,k)] for p in keys)/tot for k in ['compton','photoelectric','pair','rayleigh']} if False else None
names={'c':'compton','pe':'photoelectric','pair':'pair','coh':'rayleigh'}
def fr(d,ps):
    tot=sum(d[(p,t)] for p in ps for t in names)
    return {names[t]: sum(d[(p,t)] for p in ps)/tot for t in names}
L2i=fr(inter,'PS'); L2t=fr(transf,'PS')
L2ip=fr(inter,'P'); L2is=fr(inter,'S')
print("interaction fractions",L2i); print("energy transfer",L2t)
print("primary",L2ip); print("scattered",L2is)
scat_share=sum(inter[('S',t)] for t in names)/sum(inter.values())
print("scattered share of interactions",scat_share)
print("scattered share of transfer", sum(transf[('S',t)] for t in names)/sum(transf.values()))
