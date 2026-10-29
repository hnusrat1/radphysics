
import numpy as np, glob, time
time.sleep(60)
print([open(f).read() for f in sorted(glob.glob('/tmp/work/log*.txt'))], glob.glob('/tmp/work/H_*.npz'))
