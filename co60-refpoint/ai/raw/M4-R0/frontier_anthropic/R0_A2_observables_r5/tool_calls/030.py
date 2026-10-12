
s = open('/tmp/work/pdd2.py').read()
s = s.replace("""    cb = np.array([-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9])
    for i,T in enumerate(cF):
        if T<0.0105: continue
        R=CSDA(T)
        s=np.linspace(0,R,ns+1); sm=0.5*(s[1:]+s[:-1]); ds=s[1]-s[0]
        Rrem=np.maximum(R-sm,1e-12)
        Eloc=np.exp(np.interp(np.log(Rrem),np.log(RANGE),np.log(TE)))
        dep=S_col(Eloc)*ds
        for j,c in enumerate(cb):
            w=C_nc[i,j]
            if w<=0: continue
            x=c*sm
            idx=np.floor((x-xmin)/zbin).astype(int)
            ok=(idx>=0)&(idx<nb)
            np.add.at(kern, idx[ok], w*dep[ok])""",
"""    sub=10
    cfine = (np.arange(sub)+0.5)/sub*0.2-0.1      # offsets within each cos bin
    for i,T in enumerate(cF):
        if T<0.0105: continue
        R=CSDA(T)
        s=np.linspace(0,R,ns+1); sm=0.5*(s[1:]+s[:-1]); ds=s[1]-s[0]
        Rrem=np.maximum(R-sm,1e-12)
        Eloc=np.exp(np.interp(np.log(Rrem),np.log(RANGE),np.log(TE)))
        dep=S_col(Eloc)*ds
        for j in range(10):
            w=C_nc[i,j]
            if w<=0: continue
            cbin=-0.9+0.2*j
            for cc in cbin+0.1+cfine:
                x=cc*sm
                idx=np.floor((x-xmin)/zbin).astype(int)
                ok=(idx>=0)&(idx<nb)
                np.add.at(kern, idx[ok], (w/sub)*dep[ok])""")
s = s.replace("ns=3000","ns=1200")
s = s.replace("""    print("D first bins/Dmax", np.round(D[:15]/D.max(),3).tolist())""",
"""    print("D first bins/Dmax", np.round(D[:15]/D.max(),3).tolist())
    print("D/K ratio", np.round(D[:30]/np.array(res['Ktot'])[:30]/ (D[-1]/res['Ktot'][-1]),3).tolist())""")
open('/tmp/work/pdd2.py','w').write(s)
import subprocess
r=subprocess.run(['python3','/tmp/work/pdd2.py'],capture_output=True,text=True,cwd='/tmp/work')
print(r.stdout[-3000:], r.stderr[-3000:])
