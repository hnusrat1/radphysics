
s=open('/tmp/work/mcph2.py').read()
old=s[s.index("            # Q slabs (radius 3 cm)"):s.index("        # interactions: process")]
new_block='''        # Q slabs (radius 3 cm), over all photon segments
        Lr_all,Llo_all,Lhi_all=chord(P,D,move,3.0,-1e9,1e9)
        okq=Lr_all>0
        if okq.any():
            Ep_=E[okq]; Dz_=D[okq,2]; Pz_=P[okq,2]
            wq_all=(Ep_*mc_['inc'][okq]*fkn(Ep_)+mc_['pe'][okq]*Ep_+mc_['pr'][okq]*np.maximum(Ep_-1.022,0))
            zs_=Pz_+Llo_all[okq]*Dz_; ze_=Pz_+Lhi_all[okq]*Dz_
            zlo_=np.minimum(zs_,ze_); zhi_=np.maximum(zs_,ze_)
            ok2=np.abs(Dz_)>1e-9
            jlo_=np.clip(np.floor(zlo_/0.25).astype(int),0,nbins_q-1)
            jhi_=np.clip(np.floor(zhi_/0.25).astype(int),0,nbins_q-1)
            for j in range(nbins_q):
                za=j*0.25; zb=za+0.25
                m_=ok2&(jhi_>=j)&(jlo_<=j)
                if not m_.any(): continue
                ov=np.minimum(zhi_[m_],zb)-np.maximum(zlo_[m_],za)
                Qslab[j]+=np.sum(np.maximum(ov,0)/np.abs(Dz_[m_])*wq_all[m_])
'''
s=s.replace(old,new_block)
open('/tmp/work/mcph2.py','w').write(s)
import sys; sys.path.insert(0,'/tmp/work')
import importlib, mcph2
r=mcph2.run(200000,5)
print(np.round(r['Qslab'][:12]/(200000*np.pi*9*0.25*0.0099751**-1*0+1)/(np.pi*9*0.25)/(1/(0.0099751*110**2)),4))
