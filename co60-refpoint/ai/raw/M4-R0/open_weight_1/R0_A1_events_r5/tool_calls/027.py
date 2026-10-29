import json, sys
# Write the complete script to a file
script = """
import numpy as np
from scipy.interpolate import interp1d
import json, sys

np.random.seed(42)

m_e = 0.51099895
r_e = 2.8179403262e-13
NA = 6.02214076e23
SSD = 100.0
fh = 5.0
zs = 0.0
zc = 5.0
zh = 0.1
zlo = zc - zh
zhi = zc + zh
Rc = 1.0
th = np.arctan(fh/SSD)
Opyr = 4.0 * np.arcsin(np.sin(th)**2)
ne = 10.0 * NA / 18.015

print(f"O={Opyr:.6f} ne={ne:.4e}", file=sys.stderr)

ea = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
xT = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])
xC = np.array([0.06950, 0.06650, 0.06430, 0.06350, 0.06200, 0.06060, 0.06000, 0.05850, 0.05550])
xP = np.array([0.00040, 0.00030, 0.00025, 0.00022, 0.00020, 0.00017, 0.00015, 0.00013, 0.00010])
xPP = np.array([0.0, 0.0, 0.00020, 0.00028, 0.00050, 0.00080, 0.00100, 0.00130, 0.00150])
xR = np.array([0.00080, 0.00068, 0.00060, 0.00055, 0.00050, 0.00043, 0.00040, 0.00035, 0.00030])

fT = interp1d(ea, xT, kind='cubic', bounds_error=False, fill_value='extrapolate')
"""

# Let me just write the whole thing very carefully
with open('/tmp/sim.py', 'w') as f:
    lines = [
        'import numpy as np',
        'from scipy.interpolate import interp1d',
        'import json, sys',
        '',
        'np.random.seed(42)',
        '',
        'm_e = 0.51099895',
        'r_e = 2.8179403262e-13',
    ]