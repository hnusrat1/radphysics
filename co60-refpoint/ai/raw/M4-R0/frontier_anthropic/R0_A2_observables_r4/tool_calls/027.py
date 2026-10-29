
import re as _re
src=open('/tmp/work/ana2.py').read()
src=src.replace("/tmp/work/SM.pkl","/tmp/work/SM2.pkl")
src=_re.sub(r"SM\['(hE|g|gE|hC|int|trE|kcol|kgross)'\]\[4\]", lambda m: "SM['%s'][10]"%m.group(1), src)
src=src.replace("SM['kcol'][4]","SM['kcol'][10]")
open('/tmp/work/ana2.py','w').write(src)
src3=open('/tmp/work/ana3.py').read()
src3=_re.sub(r"SM\['(hE|g|gE|hC|int|trE|kcol|kgross)'\]\[4\]", lambda m: "SM['%s'][10]"%m.group(1), src3)
src3=src3.replace("SM['kcol'][4]","SM['kcol'][10]")
# depth list for Kd: use all depths from mc3
src3=src3.replace("depth_list=[0.25,0.5,0.75,1.0,5.0]","depth_list=[0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0]")
src3=src3.replace("    Kd.append(pz+SM['kcol'][j]*F)","    Kd.append(pz+SM['kcol'][j]*F)")
open('/tmp/work/ana3.py','w').write(src3)
import re
print(re.findall(r"SM\['[a-zA-Z0-9_]+'\]\[\d+\]", src+src3))
