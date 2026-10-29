hdr_el = []
with open('/tmp/data/estar_water.txt') as f:
    for line in f:
        if line.startswith('#'):
            hdr_el.append(line)
        else:
            break
print("ELECTRON HEADER:")
print(''.join(hdr_el))
data_el = np.loadtxt('/tmp/data/estar_water.txt', comments='#')
print("cols:", data_el.shape)
print(data_el[:8])
print(data_el[-5:])