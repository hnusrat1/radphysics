for i,l in enumerate(open('/tmp/sample.txt')):
 a=l.split()
 if a[0]=='0' and float(a[3])<0: print(i,l.strip()); break
print(open('/tmp/sample.txt').readline())