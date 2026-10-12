
import subprocess,sys,time
print(subprocess.run(['bash','-lc','ls -la /tmp/work; cat /tmp/work/log*.txt'],capture_output=True,text=True).stdout)
