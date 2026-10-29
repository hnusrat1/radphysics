for f in ['/tmp/data/xcom_water.txt','/tmp/data/estar_water.txt']:
 print('\n',f)
 print(open(f).read()[:3000])