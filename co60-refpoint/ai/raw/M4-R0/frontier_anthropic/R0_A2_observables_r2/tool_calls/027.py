
ana5 = r'''
# ---------- solve chains ----------
Qp=sources('p'); Qs=sources('s'); Qa=Qp+Qs
F0p,Fdp=chain(Qp); F0s,Fds=chain(Qs)
F0a=F0p+F0s; Fda=Fdp+Fds
Ftot=F0a+Fda
# restrict to >=10 keV (idx) for outputs
mask=np.zeros(nT,bool); mask[idx]=True
Fall=Ftot*mask
L4_fluence=Fall.sum()
L4_delta=(Fda*mask).sum()/Fall.sum()
L4_meanE=(Fall*Tcen).sum()/Fall.sum()
L4_spec=np.array([Fall[(Tcen>=edges25[k])&(Tcen<edges25[k+1])].sum() for k in range(25)])
# note bin 24 is [1.2,1.4]
# photon-based kerma
mtr_E=M.mu_tr_coef(EC)
K_scat=(Rs_fine*EC*mtr_E).sum()*CG
K_prim=(R_line*ELINES*M.mu_tr_coef(ELINES)).sum()*CG
K_tot=K_scat+K_prim
Krad_tot=(Fall*Srad).sum()*CG
Krad_s=((F0s+Fds)*mask*Srad).sum()*CG
Krad_p=((F0p+Fdp)*mask*Srad).sum()*CG
Kcol=K_tot-Krad_tot
g_yield=Krad_tot/K_tot
Ksc_col=K_scat-Krad_s
D_scat_frac=Ksc_col/Kcol
# electron-source check of K (all energies)
Kcheck=(Qa*Tcen).sum()*CG
# Spencer-Attix
Lw_D=float(np.interp(DELTA,Tcen,Lw)) if False else None
LwD=phys.estar(DELTA)[0]-phys.tail_fast(np.array([DELTA*2.0001]),DELTA)[0]*0  # placeholder
# track-end at Delta
F_at=Ftot[i10]/DT
Sw_D=phys.estar(DELTA)[0]-0.0
LwD=Lw[i10]; LaD=La[i10]
# use L at T=Delta: restricted = unrestricted (T<2Delta)
num=(Fall*Lw).sum()+DELTA*F_at*LwD
den=(Fall*La).sum()+DELTA*F_at*LaD
sw_air=num/den
# depth dose
Kslab=np.zeros((2,M.NSLAB))
Kslab=t['slab_K']*Fslab*CG
Kz=Kslab.sum(axis=0)
zc=(np.arange(M.NSLAB)+0.5)*0.1
D5=0.5*(Kz[49]+Kz[50])
Dmax=Kz.max(); kmax=int(np.argmax(Kz))
PDD5=100*D5/Dmax
# scoring-volume K from slabs for consistency
Kscore_slab=0.5*(Kz[49]+Kz[50])
print('Phi_prim',Phi_prim,'Phi_scat',Phi_scat,'L1 frac',L1_scat_frac)
print('K_tot',K_tot,'K_check(e-source)',Kcheck,'Krad',Krad_tot,'Kcol',Kcol,'g',g_yield)
print('K_scat',K_scat,'Ksc_col frac',D_scat_frac)
print('L4 fluence',L4_fluence,'delta frac',L4_delta,'meanE',L4_meanE)
print('L4 spec',L4_spec.round(5))
print('s_w,air',sw_air,'num',num,'den',den,'track end',DELTA*F_at*LwD)
print('Kz (slabs 0..12):',(Kz[:13]/Dmax).round(4))
print('Dmax at z',zc[kmax],'PDD5',PDD5,'Kscore slab',Kscore_slab,'vs point K',K_tot)
'''
s=open('/tmp/ana4.py').read()+ana5
open('/tmp/ana5.py','w').write(s)
import time
t0=time.time()
g={}
exec(s,g)
print('time',time.time()-t0)
