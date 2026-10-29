
import time,sys
t00=time.time()
src=open('/tmp/ana4.py').read()
parts=src.split('\n# ')
import re as _re
lines=src.split('\n')
# instrument: time each top-level statement roughly by executing with timers
import ast
tree=ast.parse(src)
g={}
for node in tree.body:
    t0=time.time()
    exec(compile(ast.Module([node],[]),'x','exec'),g)
    dt=time.time()-t0
    if dt>0.5: print(round(dt,2), ast.get_source_segment(src,node)[:60].replace('\n',' | '))
print('total',time.time()-t00)
