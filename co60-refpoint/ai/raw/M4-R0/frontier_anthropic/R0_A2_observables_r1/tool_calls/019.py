
import subprocess
print(subprocess.run(['bash','-lc','ps aux | grep sim.py | head; nproc; uptime'],capture_output=True,text=True).stdout)
