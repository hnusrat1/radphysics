
import subprocess
print(subprocess.run(['bash','-lc','free -m; cat /sys/fs/cgroup/memory.max 2>/dev/null'],capture_output=True,text=True).stdout)
