import numpy as np
import pandas as pd

# Load XCOM data
xcom = pd.read_csv('/tmp/data/xcom_water.txt', sep=r'\s+', header=None, names=['E_MeV', 'rayleigh', 'coherent', 'incoherent', 'photoelectric', 'pair_nuc', 'pair_el', 'total'])
# Wait, let's check columns
print(xcom.head())
print(xcom.columns.tolist())
