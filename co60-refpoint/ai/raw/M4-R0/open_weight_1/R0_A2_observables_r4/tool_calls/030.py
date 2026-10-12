import sys, numpy as np, time
sys.path.insert(0,'/tmp')
import emc

# Isolated test: uniform 1-MeV electron source in an infinite-like medium, dose = q*E0
rng=np.random.default_rng(7)
N_el=30000
# positions uniformly in cylinder radius 3 cm, z in [3.5,6.5] (relative; abs z 100..)
z0=5.0; R_src=3.0; zr=1.5
r_l=np.sqrt(rng.uniform(0,R_src**2,N_el)); phi_l=rng.uniform(0,2*np.pi,N_el)
zsel=rng.uniform(z0-zr,z0+zr,N_el)
pos=np.stack([r_l*np.cos(phi_l),r_l*np.sin(phi_l),100+zsel],axis=1)
# direction: isotropic (cos uniform)
cz=rng.uniform(-1,1,N_el); sz=np.sqrt(1-cz*cz); az=rng.uniform(0,2*np.pi,N_el)
d=np.stack([sz*np.cos(az),sz*np.sin(az),cz],axis=1)
E0=np.full(N_el,1.0)
parent=np.zeros(N_el); ispos=np.zeros(N_el,bool)
w_el = (np.pi*R_src**2*2*zr)/(N_el)  # 1 birth per unit volume -> q per volume =1 ... actually births per (cm^3 per Phi0-unit) =1*V

maxN=N_el+8*N_el
P=np.zeros((maxN,3)); Dd=np.zeros((maxN,3)); En=np.zeros(maxN)
isdelta=np.zeros(maxN,bool); isposi=np.zeros(maxN,bool); Prnt=np.zeros(maxN); Al=np.zeros(maxN,bool)
P[:N_el]=pos; Dd[:N_el]=d; En[:N_el]=E0; Al[:N_el]=True
nactive=N_el; free=list(range(N_el,maxN))
Delta=emc.Delta; V=np.pi*0.2; zc=100+z0
Srest=emc.Srest; Srad_f=emc.Srad_f; Scol_f=emc.Scol_f; Ndel=emc.Ndel; highland_sigma=emc.highland_sigma
sample_delta_energy=emc.sample_delta_energy; cos_delta_fn=emc.cos_delta_fn; rotate_e=emc.rotate_e; cyl_chord=emc.cyl_chord
sumSrest=0.0; trackend=0.0; sumChord=0.0; steps=0; it=0
while np.any(Al):
    idx=np.nonzero(Al)[0]
    Ei=En[idx]; St=Srest(Ei)+Srad_f(Ei)
    ds=np.maximum(np.minimum(0.02*Ei/np.maximum(St,1e-6),(Ei-Delta)/np.maximum(St,1e-6)),1e-6)
    p0=P[idx]; dd=Dd[idx]; p1=p0+dd*ds[:,None]
    ch=cyl_chord(p0,p1,1.0,zc,0.1)
    sumSrest+=np.sum(ch*Srest(Ei))
    sumChord+=np.sum(ch)
    lam=Ndel(Ei)*ds; nd=rng.poisson(lam); totd=int(nd.sum())
    if totd>0:
        if len(free)<totd+10:
            ext=max(totd+100,50000)
            P=np.vstack([P,np.zeros((ext,3))]); Dd=np.vstack([Dd,np.zeros((ext,3))])
            En=np.concatenate([En,np.zeros(ext)]); isdelta=np.concatenate([isdelta,np.zeros(ext,bool)])
            isposi=np.concatenate([isposi,np.zeros(ext,bool)]); Prnt=np.concatenate([Prnt,np.zeros(ext)]); Al=np.concatenate([Al,np.zeros(ext,bool)])
            free.extend(range(maxN,maxN+ext)); maxN+=ext
        reps=np.repeat(np.arange(len(idx)),nd)
        Td=sample_delta_energy(Ei[reps],len(reps),rng)
        cd=cos_delta_fn(Td,Ei[reps]); az_=rng.uniform(0,2*np.pi,len(reps))
        sd=np.sqrt(np.maximum(0,1-cd*cd)); d0=Dd[reps]
        aux=np.broadcast_to(np.array([0.,0.,1.]),(len(reps),3)).copy()
        aux[np.abs(d0[:,2])>0.9]=np.array([1.,0.,0.])
        e1=np.cross(aux,d0); n1=np.sqrt(np.sum(e1*e1,axis=1,keepdims=True)); e1=e1/np.maximum(n1,1e-12)
        e2=np.cross(d0,e1)
        dd_new=cd[:,None]*d0+sd[:,None]*(np.cos(az_)[:,None]*e1+np.sin(az_)[:,None]*e2)
        Esum=np.zeros(len(idx)); np.add.at(Esum,reps,Td)
        for j in range(len(reps)):
            fi=free.pop(); P[fi]=p0[reps[j]]; Dd[fi]=dd_new[j]; En[fi]=Td[j]
            isdelta[fi]=True; Al[fi]=True
            if fi>=nactive: nactive=fi+1
    else:
        Esum=np.zeros(len(idx))
    sig=np.array([highland_sigma(e,ds[j]) for j,e in enumerate(Ei)])
    dth=rng.normal(0,sig); dph=rng.uniform(0,2*np.pi,len(idx))
    newd=rotate_e(dd,dth,dph)
    E_new=Ei-ds*(Srest(Ei)+Srad_f(Ei))-Esum
    P[idx]=p1; Dd[idx]=newd; En[idx]=E_new
    outside=(P[idx,2]<100.0)|(P[idx,2]>130.0)|(np.abs(P[idx,0])>15.0)|(np.abs(P[idx,1])>15.0)
    below=En[idx]<Delta
    in_vox=(np.abs(P[idx,2]-zc)<0.1)&(P[idx,0]**2+P[idx,1]**2<1.0)
    trackend+=np.sum(En[idx][below&in_vox])
    dead=outside|below
    Al[idx[dead]]=False
    it+=1; steps+=len(idx)
D=(sumSrest+trackend)/V*w_el
# expected: q*E0 per unit mass = (births per volume per "Phi0") * E0 = 1 * 1.0 MeV/g (since q=1 per cm^3, dose per unit mass = q*E0 = 1 MeV/g)
print("D per unit source strength (MeV/g):",D)
print("expected 1.0 MeV/g")
print("trackend fraction:",trackend/(sumSrest+trackend))
print("delta fraction of steps:",None)