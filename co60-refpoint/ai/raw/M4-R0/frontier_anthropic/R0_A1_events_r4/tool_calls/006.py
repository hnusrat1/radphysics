
code2 = open('mc.py').read()
code2 = code2.split("def main():")[0] + r'''
KEEP=200
def run_chunk(n):
    n_acc=0; txs=[];tys=[]
    while n_acc<n:
        M=int(n*1.2)+100
        tx=rng.uniform(-0.05,0.05,M); ty=rng.uniform(-0.05,0.05,M)
        acc=rng.uniform(0,1,M)<(1+tx*tx+ty*ty)**(-1.5)
        txs.append(tx[acc]); tys.append(ty[acc]); n_acc+=acc.sum()
    tx=np.concatenate(txs)[:n]; ty=np.concatenate(tys)[:n]
    nrm=np.sqrt(tx*tx+ty*ty+1)
    d=np.stack([tx/nrm,ty/nrm,1/nrm],axis=1)
    pos=np.stack([tx*100.0,ty*100.0,np.zeros(n)],axis=1)
    E=np.where(rng.uniform(0,1,n)<0.5,1.1732,1.3325)
    prim=np.ones(n,dtype=bool)
    qsum=0.0; qprim=0.0; ngen=0
    for k in C: C[k].clear()
    while E.size>0 and ngen<300:
        ngen+=1
        a,b=nee(E,pos,d,prim); qsum+=a; qprim+=b
        sK,coh,pe,pr=mu_parts(E); mu=sK+coh+pe+pr
        s=-np.log(rng.uniform(0,1,E.size))/mu
        alive=s<s_exit_box(pos,d)
        E,d,pos,sK,coh,pe,pr,mu,s=E[alive],d[alive],pos[alive],sK[alive],coh[alive],pe[alive],pr[alive],mu[alive],s[alive]
        if E.size==0: break
        pos=pos+s[:,None]*d
        u=rng.uniform(0,1,E.size)
        isC=u<sK/mu
        isR=(~isC)&(u<(sK+coh)/mu)
        isP=(~isC)&(~isR)&(u<(sK+coh+pe)/mu)
        isPair=(~isC)&(~isR)&(~isP)
        newE=[];newd=[];newp=[]
        if isC.any():
            i=np.nonzero(isC)[0]
            c=sample_KN(E[i]); Ep=E[i]/(1+E[i]/me*(1-c))
            nd=rotate(d[i],c,rng.uniform(0,2*np.pi,i.size))
            kk=Ep>=EC
            newE.append(Ep[kk]); newd.append(nd[kk]); newp.append(pos[i][kk])
        if isR.any():
            i=np.nonzero(isR)[0]
            c=sample_thomson(i.size)
            newE.append(E[i]); newd.append(rotate(d[i],c,rng.uniform(0,2*np.pi,i.size))); newp.append(pos[i])
        if isPair.any():
            i=np.nonzero(isPair)[0]; dd=iso_dirs(i.size)
            newE+=[np.full(i.size,0.511)]*2; newd+=[dd,-dd]; newp+=[pos[i]]*2
        if len(newE)==0: break
        E=np.concatenate(newE); d=np.concatenate(newd); pos=np.concatenate(newp)
        prim=np.zeros(E.size,dtype=bool)
    return qsum,qprim,ngen

def main():
    t0=time.time()
    tot_q=0.0; tot_qp=0.0; procW=np.zeros(4); procWp=np.zeros(4)
    top=None  # dict of arrays of top KEEP candidates
    nch=N0//CH
    for ch in range(nch):
        for k in C: C[k].clear()
        qs,qp,ng=run_chunk(CH)
        tot_q+=qs; tot_qp+=qp
        Cc={k:np.concatenate(v) for k,v in C.items()}
        w=Cc['w']; good=w>0
        for k in Cc: Cc[k]=Cc[k][good]; 
        w=Cc['w']
        for p in range(4):
            m=Cc['proc']==p
            procW[p]+=w[m].sum(); procWp[p]+=w[m&Cc['prim']].sum()
        lk=np.log(rng.uniform(0,1,w.size))/w
        # keep top KEEP
        if top is None:
            allk=lk; allc=Cc
        else:
            allk=np.concatenate([top[0],lk]); allc={k:np.concatenate([top[1][k],Cc[k]]) for k in Cc}
        sel=np.argsort(-allk)[:KEEP]
        top=(allk[sel],{k:allc[k][sel] for k in allc})
    print('N0',N0,'sum_w_per_source',tot_q/N0,'prim',tot_qp/N0,'procW',procW/N0,'procWprim',procWp/N0,'time',time.time()-t0,file=sys.stderr)
    np.savez('sample_%d.npz'%SEED, **top[1])

CH=200000
main()
'''
# replace the N0 global usage: N0 from argv used as total
open('mc2.py','w').write(code2)
import subprocess,time
t=time.time()
r=subprocess.run(['python','mc2.py','7','2000000'],capture_output=True,text=True)
print(r.stdout, r.stderr[-3000:], time.time()-t)
