
sim=open('/tmp/work/sim.py').read()
sim=sim.replace("""    uv=[]
    while len(uv)<N:
        u=rng.uniform(-0.05,0.05,2*N); v=rng.uniform(-0.05,0.05,2*N)
        acc=rng.uniform(0,1,2*N)< (1+u*u+v*v)**-1.5
        uv.append(np.stack([u[acc],v[acc]],1))
    uv=np.concatenate(uv)[:N]""","""    uv=np.zeros((0,2)); 
    while len(uv)<N:
        u=rng.uniform(-0.05,0.05,2*N); v=rng.uniform(-0.05,0.05,2*N)
        acc=rng.uniform(0,1,2*N)< (1+u*u+v*v)**-1.5
        uv=np.concatenate([uv,np.stack([u[acc],v[acc]],1)])
    uv=uv[:N]""")
open('/tmp/work/sim.py','w').write(sim)
r=subprocess.run([sys.executable,'/tmp/work/sim.py','20000','1','test'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout, r.stderr[-3000:])
