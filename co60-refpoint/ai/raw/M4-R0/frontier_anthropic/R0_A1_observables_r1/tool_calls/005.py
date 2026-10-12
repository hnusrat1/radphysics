
import os,sys
print(os.getcwd(), os.listdir('.')[:20])
sys.path.insert(0,os.getcwd())
open('phys.py','w').write(open('/dev/null').read() or "")
