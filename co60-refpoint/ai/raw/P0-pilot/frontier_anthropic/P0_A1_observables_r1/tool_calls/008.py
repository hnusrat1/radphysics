
mc=open('mc.py').read()
mc=mc.replace("""    c_ids=ii[kind==0]
    if c_ids.size:
        Ein=E[c_ids]
        Ep_,cs=sample_KN(Ein,rng)
        phi=rng.uniform(0,2*np.pi,c_ids.size)
        u,vv=perp_basis(d[c_ids]); sn=np.sqrt(np.clip(1-cs**2,0,1))
        d[c_ids]=cs[:,None]*d[c_ids]+(sn*np.cos(phi))[:,None]*u+(sn*np.sin(phi))[:,None]*vv
        E[c_ids]=Ep_""","""    c_ids=ii[kind==0]
    if c_ids.size:
        Ein=E[c_ids]
        Ep_,cs=sample_KN(Ein,rng)
        phi=rng.uniform(0,2*np.pi,c_ids.size)
        u,vv=perp_basis(d[c_ids]); sn=np.sqrt(np.clip(1-cs**2,0,1))
        dsc=cs[:,None]*d[c_ids]+(sn*np.cos(phi))[:,None]*u+(sn*np.sin(phi))[:,None]*vv
        pev=Ein[:,None]*d[c_ids]-Ep_[:,None]*dsc
        pev=pev/np.linalg.norm(pev,axis=1)[:,None]
        rr=np.hypot(pos[c_ids,0],pos[c_ids,1]); msk=rr<3.0
        evT['x'].append(pos[c_ids][msk]); evT['T'].append((Ein-Ep_)[msk]); evT['u'].append(pev[msk])
        d[c_ids]=dsc
        E[c_ids]=Ep_""")
mc=mc.replace("""    p_ids=ii[kind==1]; alive[p_ids]=False""","""    p_ids=ii[kind==1]; alive[p_ids]=False
    if p_ids.size:
        rr=np.hypot(pos[p_ids,0],pos[p_ids,1]); msk=rr<3.0
        evT['x'].append(pos[p_ids][msk]); evT['T'].append(E[p_ids][msk]); evT['u'].append(d[p_ids][msk])""")
mc=mc.replace("""    if a_ids.size:
        alive[a_ids]=False
        nn=a_ids.size""","""    if a_ids.size:
        alive[a_ids]=False
        nn=a_ids.size
        rr=np.hypot(pos[a_ids,0],pos[a_ids,1]); msk=rr<3.0
        xi2=rng.uniform(size=nn); totE=E[a_ids]-1.022
        for frac in [xi2,1-xi2]:
            evT['x'].append(pos[a_ids][msk]); evT['T'].append((totE*frac)[msk]); evT['u'].append(d[a_ids][msk])""")
mc=mc.replace("ep=dict(T=[],cos=[],w=[],prim=[],kind=[])","ep=dict(T=[],cos=[],w=[],prim=[],kind=[])\nevT=dict(x=[],T=[],u=[])")
mc=mc.replace("ep={k:np.concatenate(v) for k,v in ep.items()},","ep={k:np.concatenate(v) for k,v in ep.items()},ev={k:np.concatenate(v) for k,v in evT.items()},")
open('mc.py','w').write(mc)
import subprocess
r=subprocess.run(['python','mc.py','3','200000'],capture_output=True,text=True)
print(r.stdout[-300:],r.stderr[-1500:])
