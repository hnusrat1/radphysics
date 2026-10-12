
ana2 = r'''
exec(open('/tmp/work/ana1.py').read().split("# primary parent")[0])
# primary parent with many samples
nrep=400000
Ep_=np.repeat(lines.astype(float),nrep); dp_=np.tile([0,0,1.0],(2*nrep,1))
Wp_=np.repeat(prim_w/nrep,nrep)
Tp_,cp_,wp_=electrons(Ep_,dp_,Wp_)
Tall=np.concatenate([Ta,Tp_]); call=np.concatenate([ca,cp_]); wall=np.concatenate([wa,wp_])
Tprim=Tp_; cprim=cp_; wprim=wp_
out['L3_initial_spectrum']=spec(Tall,wall,ebE).tolist()
out['L3_initial_spectrum_parent_primary']=spec(Tprim,wprim,ebE).tolist()
out['L3_mean_initial_energy_MeV']=float(np.sum(Tall*wall)/np.sum(wall))
out['L3_mean_initial_energy_parent_primary_MeV']=float(np.sum(Tprim*wprim)/np.sum(wprim))
hc,_=np.histogram(call,bins=cb,weights=wall); out['L3_initial_direction']=(hc/hc.sum()).tolist()
'''
open('/tmp/work/ana2.py','w').write(ana2)
exec(open('/tmp/work/ana2.py').read())
print(np.round(out['L3_initial_spectrum'],4))
print(np.round(out['L3_initial_spectrum_parent_primary'],4))
print(out['L3_mean_initial_energy_MeV'], out['L3_mean_initial_energy_parent_primary_MeV'])
print(np.round(out['L3_initial_direction'],4))
