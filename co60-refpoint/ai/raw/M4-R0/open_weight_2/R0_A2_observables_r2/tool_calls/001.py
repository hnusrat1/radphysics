# Read the data files to understand their structure
import numpy as np

# Read photon interaction coefficients
try:
    xcom_data = np.loadtxt('/tmp/data/xcom_water.txt')
    print("XCOM Water columns:", xcom_data.shape)
    print("First few rows:")
    print(xcom_data[:5])
except Exception as e:
    print(f"Error reading XCOM: {e}")

print("\n---\n")

# Read electron stopping powers
try:
    ester_data = np.loadtxt('/tmp/data/estar_water.txt')
    print("ESTAR Water columns:", ester_data.shape)
    print("First few rows:")
    print(ester_data[:5])
except Exception as e:
    print(f"Error reading ESTAR: {e}")
