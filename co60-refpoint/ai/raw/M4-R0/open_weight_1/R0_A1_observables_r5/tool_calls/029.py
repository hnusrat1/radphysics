import sys,os
sys.path.insert(0,'/tmp/work'); os.chdir('/tmp/work')
s=open('mcbeam.py').read()
i=s.find('has_trk=trk>0')
print(s[i-200:i+800])