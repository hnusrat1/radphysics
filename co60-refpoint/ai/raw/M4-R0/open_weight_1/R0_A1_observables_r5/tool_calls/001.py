
import importlib
mods = ['xraylib','periodictable','astropy','numba','pandas','matplotlib','numpy','scipy','pyne','openmc']
for m in mods:
    try:
        importlib.import_module(m)
        print(m, 'OK')
    except Exception as e:
        print(m, 'NO:', type(e).__name__)
import numpy as np
print('numpy', np.__version__)
import scipy
print('scipy', scipy.__version__)
