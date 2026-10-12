
import sys, os
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
src = open('mcbeam.py').read()
# insert debug prints
src = src.replace("    niter=0\n    while alive.any():",
 "    niter=0\n    tot_int=0; tot_incyl=0; tot_insrc=0\n    while alive.any():")
src = src.replace("        n_int=int(nxt.sum())\n        if n_int>0:",
 "        n_int=int(nxt.sum()); tot_int+=n_int\n        if niter<=1 and n_alive>0:\n            print('iter',niter,'alive_start',n_alive,'n_int',n_int,'over',int(over.sum()))\n        if n_int>0:")
src = src.replace("            insrc=(zint>=zsrc_lo)&(zint<=zsrc_hi)&(xint**2+yint**2<=src_r**2)",
 "            insrc=(zint>=zsrc_lo)&(zint<=zsrc_hi)&(xint**2+yint**2<=src_r**2)\n            tot_incyl+=int(incyl.sum()); tot_insrc+=int(insrc.sum())")
src = src.replace("    if store_src:\n        src=dict(E=np.concatenate(srcE),",
 "    print('total ints',tot_int,'incyl',tot_incyl,'insrc',tot_insrc, 'srcE len', len(srcE))\n    if store_src:\n        src=dict(E=np.concatenate(srcE),")
open('mcbeam_debug.py','w').write(src)
print('written debug')
