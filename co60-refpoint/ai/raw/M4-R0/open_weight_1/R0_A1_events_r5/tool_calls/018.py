import numpy as np
from scipy.interpolate import interp1d
import json

np.random.seed(42)

# ============ ALL CONSTANTS ============
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
Opyr = 4*np.arcsin(np.sin(th)**2)
ne = 10.0*NA/18.015

ea = np.array([1.0, 1.1, 1.1732, 1.2, 1.25, 1.3, 1.3325, 1.4, 1.5])
mt = np.array([0.07070, 0.06780, 0.06590, 0.06520, 0.06400, 0.06280, 0.06220, 0.06080, 0.05780])