with open('/tmp/data/xcom_water.txt') as fh:
    lines=fh.readlines()
print('XCOM lines:', len(lines))
print(''.join(lines[-15:]))
print('='*80)
with open('/tmp/data/estar_water.txt') as fh:
    lines=fh.readlines()
print('ESTAR lines:', len(lines))
print(''.join(lines[-25:]))