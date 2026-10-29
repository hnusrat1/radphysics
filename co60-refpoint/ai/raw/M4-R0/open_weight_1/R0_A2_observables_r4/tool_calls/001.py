import numpy as np
hdr_ph = []
with open('/tmp/data/xcom_water.txt') as f:
    for line in f:
        if line.startswith('#'):
            hdr_ph.append(line)
        else:
            break
print("PHOTON HEADER:")
print(''.join(hdr_ph))
data_ph = np.loadtxt('/tmp/data/xcom_water.txt', comments='#')
print("cols:", data_ph.shape)
print(data_ph[:5])
print(data_ph[-5:])